#!/usr/bin/env python

"""A tool for extracting a list of symbols to export

When exporting symbols from a dll or exe we either need to mark the symbols in
the source code as __declspec(dllexport) or supply a list of symbols to the
linker. This program automates the latter by inspecting the symbol tables of a
list of link inputs and deciding which of those symbols need to be exported.

We can't just export all the defined symbols, as there's a limit of 65535
exported symbols and in clang we go way over that, particularly in a debug
build. Therefore a large part of the work is pruning symbols either which can't
be imported, or which we think are things that have definitions in public header
files (i.e. template instantiations) and we would get defined in the thing
importing these symbols anyway.
"""

from __future__ import print_function
import sys
import re
import os
import subprocess
import multiprocessing
import argparse

# Define functions which extract a list of symbols from a library using several
# different tools. We use subprocess.Popen and yield a symbol at a time instead
# of using subprocess.check_output and returning a list as, especially on
# Windows, waiting for the entire output to be ready can take a significant
# amount of time.

def dumpbin_get_symbols(lib):
    process = subprocess.Popen(['dumpbin','/symbols',lib], bufsize=1,
                               stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                               universal_newlines=True)
    process.stdin.close()
    for line in process.stdout:
        # Look for external symbols that are defined in some section
        match = re.match("^.+SECT.+External\s+\|\s+(\S+).*$", line)
        if match:
            yield match.group(1)
    process.wait()

def nm_get_symbols(lib):
    process = subprocess.Popen(['nm','-P',lib], bufsize=1,
                               stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                               universal_newlines=True)
    process.stdin.close()
    for line in process.stdout:
        # Look for external symbols that are defined in some section
        match = re.match("^(\S+)\s+[BDGRSTVW]\s+\S+\s+\S+$", line)
        if match:
            yield match.group(1)
    process.wait()

def readobj_get_symbols(lib):
    process = subprocess.Popen(['llvm-readobj','-symbols',lib], bufsize=1,
                               stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                               universal_newlines=True)
    process.stdin.close()
    for line in process.stdout:
        # When looking through the output of llvm-readobj we expect to see Name,
        # Section, then StorageClass, so record Name and Section when we see
        # them and decide if this is a defined external symbol when we see
        # StorageClass.
        match = re.search('Name: (\S+)', line)
        if match:
            name = match.group(1)
        match = re.search('Section: (\S+)', line)
        if match:
            section = match.group(1)
        match = re.search('StorageClass: (\S+)', line)
        if match:
            storageclass = match.group(1)
            if section != 'IMAGE_SYM_ABSOLUTE' and \
               section != 'IMAGE_SYM_UNDEFINED' and \
               storageclass == 'External':
                yield name
    process.wait()

# Define functions which determine if the target is 32-bit Windows (as that's
# where calling convention name decoration happens).

def dumpbin_is_32bit_windows(lib):
    # dumpbin /headers can output a huge amount of data (>100MB in a debug
    # build) so we read only up to the 'machine' line then close the output.
    process = subprocess.Popen(['dumpbin','/headers',lib], bufsize=1,
                               stdout=subprocess.PIPE, stdin=subprocess.PIPE,
                               universal_newlines=True)
    process.stdin.close()
    retval = False
    for line in process.stdout:
        match = re.match('.+machine \((\S+)\)', line)
        if match:
            retval = (match.group(1) == 'x86')
            break
    process.stdout.close()
    process.wait()
    return retval

def objdump_is_32bit_windows(lib):
    output = subprocess.check_output(['objdump','-f',lib],
                                     universal_newlines=True)
    for line in output:
        match = re.match('.+file format (\S+)', line)
        if match:
            return (match.group(1) == 'pe-i386')
    return False

def readobj_is_32bit_windows(lib):
    output = subprocess.check_output(['llvm-readobj','-file-headers',lib],
                                     universal_newlines=True)
    for line in output:
        match = re.match('Format: (\S+)', line)
        if match:
            return (match.group(1) == 'COFF-i386')
    return False

# MSVC mangles names to ?<identifier_mangling>@<type_mangling>. By examining the
# identifier/type mangling we can decide which symbols could possibly be
# required and which we can discard.
def should_keep_microsoft_symbol(symbol, calling_convention_decoration):
    # Keep unmangled (i.e. extern "C") names
    if not '?' in symbol:
        if calling_convention_decoration:
            # Remove calling convention decoration from names
            match = re.match('[_@]([^@]+)', symbol)
            if match:
                return match.group(1)
        return symbol
    # Function template instantiations start with ?$; keep the instantiations of
    # clang::Type::getAs, as some of them are explipict specializations that are
    # defined in clang's lib/AST/Type.cpp; discard the rest as it's assumed that
    # the definition is public
    elif re.match('\?\?\$getAs@.+@Type@clang@@', symbol):
        return symbol
    elif symbol.startswith('??$'):
        return None
    # Deleting destructors start with ?_G or ?_E and can be discarded because
    # link.exe gives you a warning telling you they can't be exported if you
    # don't
    elif symbol.startswith('??_G') or symbol.startswith('??_E'):
        return None
    # Constructors (?0) and destructors (?1) of templates (?$) are assumed to be
    # defined in headers and not required to be kept
    elif symbol.startswith('??0?$') or symbol.startswith('??1?$'):
        return None
    # An anonymous namespace is mangled as ?A(maybe hex number)@. Any symbol
    # that mentions an anonymous namespace can be discarded, as the anonymous
    # namespace doesn't exist outside of that translation unit.
    elif re.search('\?A(0x\w+)?@', symbol):
        return None
    # Keep mangled llvm:: and clang:: function symbols. How we detect these is a
    # bit of a mess and imprecise, but that avoids having to completely demangle
    # the symbol name. The outermost namespace is at the end of the identifier
    # mangling, and the identifier mangling is followed by the type mangling, so
    # we look for (llvm|clang)@@ followed by something that looks like a
    # function type mangling. To spot a function type we use (this is derived
    # from clang/lib/AST/MicrosoftMangle.cpp):
    # <function-type> ::= <function-class> <this-cvr-qualifiers>
    #                     <calling-convention> <return-type>
    #                     <argument-list> <throw-spec>
    # <function-class> ::= [A-Z]
    # <this-cvr-qualifiers> ::= [A-Z0-9_]*
    # <calling-convention> ::= [A-JQ]
    # <return-type> ::= .+
    # <argument-list> ::= X   (void)
    #                 ::= .+@ (list of types)
    #                 ::= .*Z (list of types, varargs)
    # <throw-spec> ::= exceptions are not allowed
    elif re.search('(llvm|clang)@@[A-Z][A-Z0-9_]*[A-JQ].+(X|.+@|.*Z)$', symbol):
        return symbol
    return None

# Itanium manglings are of the form _Z<identifier_mangling><type_mangling>. We
# demangle the identifier mangling to identify symbols that can be safely
# discarded.
def should_keep_itanium_symbol(symbol, calling_convention_decoration):
    # Start by removing any calling convention decoration (which we expect to
    # see on all symbols, even mangled C++ symbols)
    if calling_convention_decoration and symbol.startswith('_'):
        symbol = symbol[1:]
    # Keep unmangled names
    if not symbol.startswith('_') and not symbol.startswith('.'):
        return symbol
    # Discard manglings that aren't nested names
    match = re.match('_Z(T[VTIS])?(N.+)', symbol)
    if not match:
        return None
    # Demangle the name. If the name is too complex then we don't need to keep
    # it, but it the demangling fails then keep the symbol just in case.
    try:
        names, _ = parse_itanium_nested_name(match.group(2))
    except TooComplexName:
        return None
    if not names:
        return symbol
    # Constructors and destructors of templates classes are assumed to be
    # defined in headers and not required to be kept
    if re.match('[CD][123]', names[-1][0]) and names[-2][1]:
        return None
    # Keep the instantiations of clang::Type::getAs, as some of them are
    # explipict specializations that are defined in clang's lib/AST/Type.cpp;
    # discard any other function template instantiations as it's assumed that
    # the definition is public
    elif symbol.startswith('_ZNK5clang4Type5getAs'):
        return symbol
    elif names[-1][1]:
        return None
    # Keep llvm:: and clang:: names
    elif names[0][0] == '4llvm' or names[0][0] == '5clang':
        return symbol
    # Discard everything else
    else:
        return None

# Certain kinds of complex manglings we assume cannot be part of a public
# interface, and we handle them by raising an exception.
class TooComplexName(Exception):
    pass

# Parse an itanium mangled name from the start of a string and return a
# (name, rest of string) pair.
def parse_itanium_name(arg):
    # Check for a normal name
    match = re.match('(\d+)(.+)', arg)
    if match:
        n = int(match.group(1))
        name = match.group(1)+match.group(2)[:n]
        rest = match.group(2)[n:]
        return name, rest
    # Check for constructor/destructor names
    match = re.match('([CD][123])(.+)', arg)
    if match:
        return match.group(1), match.group(2)
    # Assume that a sequence of characters that doesn't end a nesting is an
    # operator (this is very imprecise, but appears to be good enough)
    match = re.match('([^E]+)(.+)', arg)
    if match:
        return match.group(1), match.group(2)
    # Anything else: we can't handle it
    return None, arg

# Parse an itanium mangled template argument list from the start of a string
# and throw it away, returning the rest of the string.
def skip_itanium_template(arg):
    # A template argument list starts with I
    assert arg.startswith('I'), arg
    tmp = arg[1:]
    while tmp:
        # Check for names
        match = re.match('(\d+)(.+)', tmp)
        if match:
            n = int(match.group(1))
            tmp =  match.group(2)[n:]
            continue
        # Check for substitutions
        match = re.match('S[A-Z0-9]*_(.+)', tmp)
        if match:
            tmp = match.group(1)
        # Start of a template
        elif tmp.startswith('I'):
            tmp = skip_itanium_template(tmp)
        # Start of a nested name
        elif tmp.startswith('N'):
            _, tmp = parse_itanium_nested_name(tmp)
        # Start of an expression: assume that it's too complicated
        elif tmp.startswith('L') or tmp.startswith('X'):
            raise TooComplexName
        # End of the template
        elif tmp.startswith('E'):
            return tmp[1:]
        # Something else: probably a type, skip it
        else:
            tmp = tmp[1:]
    return None

# Parse an itanium mangled nested name and transform it into a list of pairs of
# (name, is_template), returning (list, rest of string).
def parse_itanium_nested_name(arg):
    # A nested name starts with N
    assert arg.startswith('N'), arg
    ret = []

    # Skip past the N, and possibly a substitution
    match = re.match('NS[A-Z0-9]*_(.+)', arg)
    if match:
        tmp = match.group(1)
    else:
        tmp = arg[1:]

    # Skip past CV-qualifiers and ref qualifiers
    match = re.match('[rVKRO]*(.+)', tmp);
    if match:
        tmp = match.group(1)

    # Repeatedly parse names from the string until we reach the end of the
    # nested name
    while tmp:
        # An E ends the nested name
        if tmp.startswith('E'):
            return ret, tmp[1:]
        # Parse a name
        name_part, tmp = parse_itanium_name(tmp)
        if not name_part:
            # If we failed then we don't know how to demangle this
            return None, None
        is_template = False
        # If this name is a template record that, then skip the template
        # arguments
        if tmp.startswith('I'):
            tmp = skip_itanium_template(tmp)
            is_template = True
        # Add the name to the list
        ret.append((name_part, is_template))

    # If we get here then something went wrong
    return None, None

def extract_symbols(arg):
    get_symbols, should_keep_symbol, calling_convention_decoration, lib = arg
    symbols = dict()
    for symbol in get_symbols(lib):
        symbol = should_keep_symbol(symbol, calling_convention_decoration)
        if symbol:
            symbols[symbol] = 1 + symbols.setdefault(symbol,0)
    return symbols

if __name__ == '__main__':
    tool_exes = ['dumpbin','nm','objdump','llvm-readobj']
    parser = argparse.ArgumentParser(
        description='Extract symbols to export from libraries')
    parser.add_argument('--mangling', choices=['itanium','microsoft'],
                        required=True, help='expected symbol mangling scheme')
    parser.add_argument('--tools', choices=tool_exes, nargs='*',
                        help='tools to use to extract symbols and determine the'
                        ' target')
    parser.add_argument('libs', metavar='lib', type=str, nargs='+',
                        help='libraries to extract symbols from')
    parser.add_argument('-o', metavar='file', type=str, help='output to file')
    args = parser.parse_args()

    # Determine the function to use to get the list of symbols from the inputs,
    # and the function to use to determine if the target is 32-bit windows.
    tools = { 'dumpbin' : (dumpbin_get_symbols, dumpbin_is_32bit_windows),
              'nm' : (nm_get_symbols, None),
              'objdump' : (None, objdump_is_32bit_windows),
              'llvm-readobj' : (readobj_get_symbols, readobj_is_32bit_windows) }
    get_symbols = None
    is_32bit_windows = None
    # If we have a tools argument then use that for the list of tools to check
    if args.tools:
        tool_exes = args.tools
    # Find a tool to use by trying each in turn until we find one that exists
    # (subprocess.call will throw OSError when the program does not exist)
    get_symbols = None
    for exe in tool_exes:
        try:
            # Close std streams as we don't want any output and we don't
            # want the process to wait for something on stdin.
            p = subprocess.Popen([exe], stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE,
                                 stdin=subprocess.PIPE,
                                 universal_newlines=True)
            p.stdout.close()
            p.stderr.close()
            p.stdin.close()
            p.wait()
            # Keep going until we have a tool to use for both get_symbols and
            # is_32bit_windows
            if not get_symbols:
                get_symbols = tools[exe][0]
            if not is_32bit_windows:
                is_32bit_windows = tools[exe][1]
            if get_symbols and is_32bit_windows:
                break
        except OSError:
            continue
    if not get_symbols:
        print("Couldn't find a program to read symbols with", file=sys.stderr)
        exit(1)
    if not is_32bit_windows:
        print("Couldn't find a program to determining the target", file=sys.stderr)
        exit(1)

    # How we determine which symbols to keep and which to discard depends on
    # the mangling scheme
    if args.mangling == 'microsoft':
        should_keep_symbol = should_keep_microsoft_symbol
    else:
        should_keep_symbol = should_keep_itanium_symbol

    # Get the list of libraries to extract symbols from
    libs = list()
    for lib in args.libs:
        # When invoked by cmake the arguments are the cmake target names of the
        # libraries, so we need to add .lib/.a to the end and maybe lib to the
        # start to get the filename. Also allow objects.
        suffixes = ['.lib','.a','.obj','.o']
        if not any([lib.endswith(s) for s in suffixes]):
            for s in suffixes:
                if os.path.exists(lib+s):
                    lib = lib+s
                    break
                if os.path.exists('lib'+lib+s):
                    lib = 'lib'+lib+s
                    break
        if not any([lib.endswith(s) for s in suffixes]):
            print("Don't know what to do with argument "+lib, file=sys.stderr)
            exit(1)
        libs.append(lib)

    # Check if calling convention decoration is used by inspecting the first
    # library in the list
    calling_convention_decoration = is_32bit_windows(libs[0])

    # Extract symbols from libraries in parallel. This is a huge time saver when
    # doing a debug build, as there are hundreds of thousands of symbols in each
    # library.
    pool = multiprocessing.Pool()
    try:
        # Only one argument can be passed to the mapping function, and we can't
        # use a lambda or local function definition as that doesn't work on
        # windows, so create a list of tuples which duplicates the arguments
        # that are the same in all calls.
        vals = [(get_symbols, should_keep_symbol, calling_convention_decoration, x) for x in libs]
        # Do an async map then wait for the result to make sure that
        # KeyboardInterrupt gets caught correctly (see
        # http://bugs.python.org/issue8296)
        result = pool.map_async(extract_symbols, vals)
        pool.close()
        libs_symbols = result.get(3600)
    except KeyboardInterrupt:
        # On Ctrl-C terminate everything and exit
        pool.terminate()
        pool.join()
        exit(1)

    # Merge everything into a single dict
    symbols = dict()
    for this_lib_symbols in libs_symbols:
        for k,v in list(this_lib_symbols.items()):
            symbols[k] = v + symbols.setdefault(k,0)

    # Count instances of member functions of template classes, and map the
    # symbol name to the function+class. We do this under the assumption that if
    # a member function of a template class is instantiated many times it's
    # probably declared in a public header file.
    template_function_count = dict()
    template_function_mapping = dict()
    template_function_count[""] = 0
    for k in symbols:
        name = None
        if args.mangling == 'microsoft':
            # Member functions of templates start with
            # ?<fn_name>@?$<class_name>@, so we map to <fn_name>@?$<class_name>.
            # As manglings go from the innermost scope to the outermost scope
            # this means:
            #  * When we have a function member of a subclass of a template
            #    class then <fn_name> will actually contain the mangling of
            #    both the subclass and the function member. This is fine.
            #  * When we have a function member of a template subclass of a
            #    (possibly template) class then it's the innermost template
            #    subclass that becomes <class_name>. This should be OK so long
            #    as we don't have multiple classes with a template subclass of
            #    the same name.
            match = re.search("^\?(\??\w+\@\?\$\w+)\@", k)
            if match:
                name = match.group(1)
        else:
            # Find member functions of templates by demangling the name and
            # checking if the second-to-last name in the list is a template.
            match = re.match('_Z(T[VTIS])?(N.+)', k)
            if match:
                try:
                    names, _ = parse_itanium_nested_name(match.group(2))
                    if names and names[-2][1]:
                        name = ''.join([x for x,_ in names])
                except TooComplexName:
                    # Manglings that are too complex should already have been
                    # filtered out, but if we happen to somehow see one here
                    # just leave it as-is.
                    pass
        if name:
            old_count = template_function_count.setdefault(name,0)
            template_function_count[name] = old_count + 1
            template_function_mapping[k] = name
        else:
            template_function_mapping[k] = ""

    # Print symbols which both:
    #  * Appear in exactly one input, as symbols defined in multiple
    #    objects/libraries are assumed to have public definitions.
    #  * Aren't instances of member functions of templates which have been
    #    instantiated 100 times or more, which are assumed to have public
    #    definitions. (100 is an arbitrary guess here.)
    if args.o:
        outfile = open(args.o,'w')
    else:
        outfile = sys.stdout
    for k,v in list(symbols.items()):
        template_count = template_function_count[template_function_mapping[k]]
        if v == 1 and template_count < 100:
            print(k, file=outfile)
