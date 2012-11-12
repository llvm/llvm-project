import StringIO
import os
import sys

import componentinfo
import configutil

from util import *

###

def cmake_quote_string(value):
    """
    cmake_quote_string(value) -> str

    Return a quoted form of the given value that is suitable for use in CMake
    language files.
    """

    # Currently, we only handle escaping backslashes.
    value = value.replace("\\", "\\\\")

    return value

def cmake_quote_path(value):
    """
    cmake_quote_path(value) -> str

    Return a quoted form of the given value that is suitable for use in CMake
    language files.
    """

    # CMake has a bug in it's Makefile generator that doesn't properly quote
    # strings it generates. So instead of using proper quoting, we just use "/"
    # style paths.  Currently, we only handle escaping backslashes.
    value = value.replace("\\", "/")

    return value

def mk_quote_string_for_target(value):
    """
    mk_quote_string_for_target(target_name) -> str

    Return a quoted form of the given target_name suitable for including in a 
    Makefile as a target name.
    """

    # The only quoting we currently perform is for ':', to support msys users.
    return value.replace(":", "\\:")

def make_install_dir(path):
    """
    make_install_dir(path) -> None

    Create the given directory path for installation, including any parents.
    """

    # os.makedirs considers it an error to be called with an existent path.
    if not os.path.exists(path):
        os.makedirs(path)

###

class LLVMProjectInfo(object):
    @staticmethod
    def load_infos_from_path(llvmbuild_source_root):
        def recurse(subpath):
            # Load the LLVMBuild file.
            llvmbuild_path = os.path.join(llvmbuild_source_root + subpath,
                                          'LLVMBuild.txt')
            if not os.path.exists(llvmbuild_path):
                fatal("missing LLVMBuild.txt file at: %r" % (llvmbuild_path,))

            # Parse the components from it.
            common,info_iter = componentinfo.load_from_path(llvmbuild_path,
                                                            subpath)
            for info in info_iter:
                yield info

            # Recurse into the specified subdirectories.
            for subdir in common.get_list("subdirectories"):
                for item in recurse(os.path.join(subpath, subdir)):
                    yield item

        return recurse("/")

    @staticmethod
    def load_from_path(source_root, llvmbuild_source_root):
        infos = list(
            LLVMProjectInfo.load_infos_from_path(llvmbuild_source_root))

        return LLVMProjectInfo(source_root, infos)

    def __init__(self, source_root, component_infos):
        # Store our simple ivars.
        self.source_root = source_root
        self.component_infos = list(component_infos)
        self.component_info_map = None
        self.ordered_component_infos = None

    def validate_components(self):
        """validate_components() -> None

        Validate that the project components are well-defined. Among other
        things, this checks that:
          - Components have valid references.
          - Components references do not form cycles.

        We also construct the map from component names to info, and the
        topological ordering of components.
        """

        # Create the component info map and validate that component names are
        # unique.
        self.component_info_map = {}
        for ci in self.component_infos:
            existing = self.component_info_map.get(ci.name)
            if existing is not None:
                # We found a duplicate component name, report it and error out.
                fatal("found duplicate component %r (at %r and %r)" % (
                        ci.name, ci.subpath, existing.subpath))
            self.component_info_map[ci.name] = ci

        # Disallow 'all' as a component name, which is a special case.
        if 'all' in self.component_info_map:
            fatal("project is not allowed to define 'all' component")

        # Add the root component.
        if '$ROOT' in self.component_info_map:
            fatal("project is not allowed to define $ROOT component")
        self.component_info_map['$ROOT'] = componentinfo.GroupComponentInfo(
            '/', '$ROOT', None)
        self.component_infos.append(self.component_info_map['$ROOT'])

        # Topologically order the component information according to their
        # component references.
        def visit_component_info(ci, current_stack, current_set):
            # Check for a cycles.
            if ci in current_set:
                # We found a cycle, report it and error out.
                cycle_description = ' -> '.join(
                    '%r (%s)' % (ci.name, relation)
                    for relation,ci in current_stack)
                fatal("found cycle to %r after following: %s -> %s" % (
                        ci.name, cycle_description, ci.name))

            # If we have already visited this item, we are done.
            if ci not in components_to_visit:
                return

            # Otherwise, mark the component info as visited and traverse.
            components_to_visit.remove(ci)

            # Validate the parent reference, which we treat specially.
            if ci.parent is not None:
                parent = self.component_info_map.get(ci.parent)
                if parent is None:
                    fatal("component %r has invalid reference %r (via %r)" % (
                            ci.name, ci.parent, 'parent'))
                ci.set_parent_instance(parent)

            for relation,referent_name in ci.get_component_references():
                # Validate that the reference is ok.
                referent = self.component_info_map.get(referent_name)
                if referent is None:
                    fatal("component %r has invalid reference %r (via %r)" % (
                            ci.name, referent_name, relation))

                # Visit the reference.
                current_stack.append((relation,ci))
                current_set.add(ci)
                visit_component_info(referent, current_stack, current_set)
                current_set.remove(ci)
                current_stack.pop()

            # Finally, add the component info to the ordered list.
            self.ordered_component_infos.append(ci)

        # FIXME: We aren't actually correctly checking for cycles along the
        # parent edges. Haven't decided how I want to handle this -- I thought
        # about only checking cycles by relation type. If we do that, it falls
        # out easily. If we don't, we should special case the check.

        self.ordered_component_infos = []
        components_to_visit = set(self.component_infos)
        while components_to_visit:
            visit_component_info(iter(components_to_visit).next(), [], set())

        # Canonicalize children lists.
        for c in self.ordered_component_infos:
            c.children.sort(key = lambda c: c.name)

    def print_tree(self):
        def visit(node, depth = 0):
            print '%s%-40s (%s)' % ('  '*depth, node.name, node.type_name)
            for c in node.children:
                visit(c, depth + 1)
        visit(self.component_info_map['$ROOT'])

    def write_components(self, output_path):
        # Organize all the components by the directory their LLVMBuild file
        # should go in.
        info_basedir = {}
        for ci in self.component_infos:
            # Ignore the $ROOT component.
            if ci.parent is None:
                continue

            info_basedir[ci.subpath] = info_basedir.get(ci.subpath, []) + [ci]

        # Compute the list of subdirectories to scan.
        subpath_subdirs = {}
        for ci in self.component_infos:
            # Ignore root components.
            if ci.subpath == '/':
                continue

            # Otherwise, append this subpath to the parent list.
            parent_path = os.path.dirname(ci.subpath)
            subpath_subdirs[parent_path] = parent_list = subpath_subdirs.get(
                parent_path, set())
            parent_list.add(os.path.basename(ci.subpath))

        # Generate the build files.
        for subpath, infos in info_basedir.items():
            # Order the components by name to have a canonical ordering.
            infos.sort(key = lambda ci: ci.name)

            # Format the components into llvmbuild fragments.
            fragments = []

            # Add the common fragments.
            subdirectories = subpath_subdirs.get(subpath)
            if subdirectories:
                fragment = """\
subdirectories = %s
""" % (" ".join(sorted(subdirectories)),)
                fragments.append(("common", fragment))

            # Add the component fragments.
            num_common_fragments = len(fragments)
            for ci in infos:
                fragment = ci.get_llvmbuild_fragment()
                if fragment is None:
                    continue

                name = "component_%d" % (len(fragments) - num_common_fragments)
                fragments.append((name, fragment))

            if not fragments:
                continue

            assert subpath.startswith('/')
            directory_path = os.path.join(output_path, subpath[1:])

            # Create the directory if it does not already exist.
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            # In an effort to preserve comments (which aren't parsed), read in
            # the original file and extract the comments. We only know how to
            # associate comments that prefix a section name.
            f = open(infos[0]._source_path)
            comments_map = {}
            comment_block = ""
            for ln in f:
                if ln.startswith(';'):
                    comment_block += ln
                elif ln.startswith('[') and ln.endswith(']\n'):
                    comments_map[ln[1:-2]] = comment_block
                else:
                    comment_block = ""
            f.close()

            # Create the LLVMBuild fil[e.
            file_path = os.path.join(directory_path, 'LLVMBuild.txt')
            f = open(file_path, "w")

            # Write the header.
            header_fmt = ';===- %s %s-*- Conf -*--===;'
            header_name = '.' + os.path.join(subpath, 'LLVMBuild.txt')
            header_pad = '-' * (80 - len(header_fmt % (header_name, '')))
            header_string = header_fmt % (header_name, header_pad)
            print >>f, """\
%s
;
;                     The LLVM Compiler Infrastructure
;
; This file is distributed under the University of Illinois Open Source
; License. See LICENSE.TXT for details.
;
;===------------------------------------------------------------------------===;
;
; This is an LLVMBuild description file for the components in this subdirectory.
;
; For more information on the LLVMBuild system, please see:
;
;   http://llvm.org/docs/LLVMBuild.html
;
;===------------------------------------------------------------------------===;
""" % header_string

            # Write out each fragment.each component fragment.
            for name,fragment in fragments:
                comment = comments_map.get(name)
                if comment is not None:
                    f.write(comment)
                print >>f, "[%s]" % name
                f.write(fragment)
                if fragment is not fragments[-1][1]:
                    print >>f

            f.close()

    def write_library_table(self, output_path, enabled_optional_components):
        # Write out the mapping from component names to required libraries.
        #
        # We do this in topological order so that we know we can append the
        # dependencies for added library groups.
        entries = {}
        for c in self.ordered_component_infos:
            # Skip optional components which are not enabled.
            if c.type_name == 'OptionalLibrary' \
                and c.name not in enabled_optional_components:
                continue

            # Skip target groups which are not enabled.
            tg = c.get_parent_target_group()
            if tg and not tg.enabled:
                continue

            # Only certain components are in the table.
            if c.type_name not in ('Library', 'OptionalLibrary', \
                                   'LibraryGroup', 'TargetGroup'):
                continue

            # Compute the llvm-config "component name". For historical reasons,
            # this is lowercased based on the library name.
            llvmconfig_component_name = c.get_llvmconfig_component_name()
            
            # Get the library name, or None for LibraryGroups.
            if c.type_name == 'Library' or c.type_name == 'OptionalLibrary':
                library_name = c.get_prefixed_library_name()
                is_installed = c.installed
            else:
                library_name = None
                is_installed = True

            # Get the component names of all the required libraries.
            required_llvmconfig_component_names = [
                self.component_info_map[dep].get_llvmconfig_component_name()
                for dep in c.required_libraries]

            # Insert the entries for library groups we should add to.
            for dep in c.add_to_library_groups:
                entries[dep][2].append(llvmconfig_component_name)

            # Add the entry.
            entries[c.name] = (llvmconfig_component_name, library_name,
                               required_llvmconfig_component_names,
                               is_installed)

        # Convert to a list of entries and sort by name.
        entries = entries.values()

        # Create an 'all' pseudo component. We keep the dependency list small by
        # only listing entries that have no other dependents.
        root_entries = set(e[0] for e in entries)
        for _,_,deps,_ in entries:
            root_entries -= set(deps)
        entries.append(('all', None, root_entries, True))

        entries.sort()

        # Compute the maximum number of required libraries, plus one so there is
        # always a sentinel.
        max_required_libraries = max(len(deps)
                                     for _,_,deps,_ in entries) + 1

        # Write out the library table.
        make_install_dir(os.path.dirname(output_path))
        f = open(output_path, 'w')
        print >>f, """\
//===- llvm-build generated file --------------------------------*- C++ -*-===//
//
// Component Library Depenedency Table
//
// Automatically generated file, do not edit!
//
//===----------------------------------------------------------------------===//
"""
        print >>f, 'struct AvailableComponent {'
        print >>f, '  /// The name of the component.'
        print >>f, '  const char *Name;'
        print >>f, ''
        print >>f, '  /// The name of the library for this component (or NULL).'
        print >>f, '  const char *Library;'
        print >>f, ''
        print >>f, '  /// Whether the component is installed.'
        print >>f, '  bool IsInstalled;'
        print >>f, ''
        print >>f, '\
  /// The list of libraries required when linking this component.'
        print >>f, '  const char *RequiredLibraries[%d];' % (
            max_required_libraries)
        print >>f, '} AvailableComponents[%d] = {' % len(entries)
        for name,library_name,required_names,is_installed in entries:
            if library_name is None:
                library_name_as_cstr = '0'
            else:
                library_name_as_cstr = '"lib%s.a"' % library_name
            print >>f, '  { "%s", %s, %d, { %s } },' % (
                name, library_name_as_cstr, is_installed,
                ', '.join('"%s"' % dep
                          for dep in required_names))
        print >>f, '};'
        f.close()

    def get_required_libraries_for_component(self, ci, traverse_groups = False):
        """
        get_required_libraries_for_component(component_info) -> iter

        Given a Library component info descriptor, return an iterator over all
        of the directly required libraries for linking with this component. If
        traverse_groups is True, then library and target groups will be
        traversed to include their required libraries.
        """

        assert ci.type_name in ('Library', 'LibraryGroup', 'TargetGroup')

        for name in ci.required_libraries:
            # Get the dependency info.
            dep = self.component_info_map[name]

            # If it is a library, yield it.
            if dep.type_name == 'Library':
                yield dep
                continue

            # Otherwise if it is a group, yield or traverse depending on what
            # was requested.
            if dep.type_name in ('LibraryGroup', 'TargetGroup'):
                if not traverse_groups:
                    yield dep
                    continue

                for res in self.get_required_libraries_for_component(dep, True):
                    yield res

    def get_fragment_dependencies(self):
        """
        get_fragment_dependencies() -> iter

        Compute the list of files (as absolute paths) on which the output
        fragments depend (i.e., files for which a modification should trigger a
        rebuild of the fragment).
        """

        # Construct a list of all the dependencies of the Makefile fragment
        # itself. These include all the LLVMBuild files themselves, as well as
        # all of our own sources.
        #
        # Many components may come from the same file, so we make sure to unique
        # these.
        build_paths = set()
        for ci in self.component_infos:
            p = os.path.join(self.source_root, ci.subpath[1:], 'LLVMBuild.txt')
            if p not in build_paths:
                yield p
                build_paths.add(p)

        # Gather the list of necessary sources by just finding all loaded
        # modules that are inside the LLVM source tree.
        for module in sys.modules.values():
            # Find the module path.
            if not hasattr(module, '__file__'):
                continue
            path = getattr(module, '__file__')
            if not path:
                continue

            # Strip off any compiled suffix.
            if os.path.splitext(path)[1] in ['.pyc', '.pyo', '.pyd']:
                path = path[:-1]

            # If the path exists and is in the source tree, consider it a
            # dependency.
            if (path.startswith(self.source_root) and os.path.exists(path)):
                yield path

    def write_cmake_fragment(self, output_path):
        """
        write_cmake_fragment(output_path) -> None

        Generate a CMake fragment which includes all of the collated LLVMBuild
        information in a format that is easily digestible by a CMake. The exact
        contents of this are closely tied to how the CMake configuration
        integrates LLVMBuild, see CMakeLists.txt in the top-level.
        """

        dependencies = list(self.get_fragment_dependencies())

        # Write out the CMake fragment.
        make_install_dir(os.path.dirname(output_path))
        f = open(output_path, 'w')

        # Write the header.
        header_fmt = '\
#===-- %s - LLVMBuild Configuration for LLVM %s-*- CMake -*--===#'
        header_name = os.path.basename(output_path)
        header_pad = '-' * (80 - len(header_fmt % (header_name, '')))
        header_string = header_fmt % (header_name, header_pad)
        print >>f, """\
%s
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
#
# This file contains the LLVMBuild project information in a format easily
# consumed by the CMake based build system.
#
# This file is autogenerated by llvm-build, do not edit!
#
#===------------------------------------------------------------------------===#
""" % header_string

        # Write the dependency information in the best way we can.
        print >>f, """
# LLVMBuild CMake fragment dependencies.
#
# CMake has no builtin way to declare that the configuration depends on
# a particular file. However, a side effect of configure_file is to add
# said input file to CMake's internal dependency list. So, we use that
# and a dummy output file to communicate the dependency information to
# CMake.
#
# FIXME: File a CMake RFE to get a properly supported version of this
# feature."""
        for dep in dependencies:
            print >>f, """\
configure_file(\"%s\"
               ${CMAKE_CURRENT_BINARY_DIR}/DummyConfigureOutput)""" % (
                cmake_quote_path(dep),)

        # Write the properties we use to encode the required library dependency
        # information in a form CMake can easily use directly.
        print >>f, """
# Explicit library dependency information.
#
# The following property assignments effectively create a map from component
# names to required libraries, in a way that is easily accessed from CMake."""
        for ci in self.ordered_component_infos:
            # We only write the information for libraries currently.
            if ci.type_name != 'Library':
                continue

            print >>f, """\
set_property(GLOBAL PROPERTY LLVMBUILD_LIB_DEPS_%s %s)""" % (
                ci.get_prefixed_library_name(), " ".join(sorted(
                     dep.get_prefixed_library_name()
                     for dep in self.get_required_libraries_for_component(ci))))

        f.close()

    def write_make_fragment(self, output_path):
        """
        write_make_fragment(output_path) -> None

        Generate a Makefile fragment which includes all of the collated
        LLVMBuild information in a format that is easily digestible by a
        Makefile. The exact contents of this are closely tied to how the LLVM
        Makefiles integrate LLVMBuild, see Makefile.rules in the top-level.
        """

        dependencies = list(self.get_fragment_dependencies())

        # Write out the Makefile fragment.
        make_install_dir(os.path.dirname(output_path))
        f = open(output_path, 'w')

        # Write the header.
        header_fmt = '\
#===-- %s - LLVMBuild Configuration for LLVM %s-*- Makefile -*--===#'
        header_name = os.path.basename(output_path)
        header_pad = '-' * (80 - len(header_fmt % (header_name, '')))
        header_string = header_fmt % (header_name, header_pad)
        print >>f, """\
%s
#
#                     The LLVM Compiler Infrastructure
#
# This file is distributed under the University of Illinois Open Source
# License. See LICENSE.TXT for details.
#
#===------------------------------------------------------------------------===#
#
# This file contains the LLVMBuild project information in a format easily
# consumed by the Makefile based build system.
#
# This file is autogenerated by llvm-build, do not edit!
#
#===------------------------------------------------------------------------===#
""" % header_string

        # Write the dependencies for the fragment.
        #
        # FIXME: Technically, we need to properly quote for Make here.
        print >>f, """\
# Clients must explicitly enable LLVMBUILD_INCLUDE_DEPENDENCIES to get
# these dependencies. This is a compromise to help improve the
# performance of recursive Make systems.""" 
        print >>f, 'ifeq ($(LLVMBUILD_INCLUDE_DEPENDENCIES),1)'
        print >>f, "# The dependencies for this Makefile fragment itself."
        print >>f, "%s: \\" % (mk_quote_string_for_target(output_path),)
        for dep in dependencies:
            print >>f, "\t%s \\" % (dep,)
        print >>f

        # Generate dummy rules for each of the dependencies, so that things
        # continue to work correctly if any of those files are moved or removed.
        print >>f, """\
# The dummy targets to allow proper regeneration even when files are moved or
# removed."""
        for dep in dependencies:
            print >>f, "%s:" % (mk_quote_string_for_target(dep),)
        print >>f, 'endif'

        f.close()

def add_magic_target_components(parser, project, opts):
    """add_magic_target_components(project, opts) -> None

    Add the "magic" target based components to the project, which can only be
    determined based on the target configuration options.

    This currently is responsible for populating the required_libraries list of
    the "all-targets", "Native", "NativeCodeGen", and "Engine" components.
    """

    # Determine the available targets.
    available_targets = dict((ci.name,ci)
                             for ci in project.component_infos
                             if ci.type_name == 'TargetGroup')

    # Find the configured native target.

    # We handle a few special cases of target names here for historical
    # reasons, as these are the names configure currently comes up with.
    native_target_name = { 'x86' : 'X86',
                           'x86_64' : 'X86',
                           'Unknown' : None }.get(opts.native_target,
                                                  opts.native_target)
    if native_target_name is None:
        native_target = None
    else:
        native_target = available_targets.get(native_target_name)
        if native_target is None:
            parser.error("invalid native target: %r (not in project)" % (
                    opts.native_target,))
        if native_target.type_name != 'TargetGroup':
            parser.error("invalid native target: %r (not a target)" % (
                    opts.native_target,))

    # Find the list of targets to enable.
    if opts.enable_targets is None:
        enable_targets = available_targets.values()
    else:
        # We support both space separated and semi-colon separated lists.
        if ' ' in opts.enable_targets:
            enable_target_names = opts.enable_targets.split()
        else:
            enable_target_names = opts.enable_targets.split(';')

        enable_targets = []
        for name in enable_target_names:
            target = available_targets.get(name)
            if target is None:
                parser.error("invalid target to enable: %r (not in project)" % (
                        name,))
            if target.type_name != 'TargetGroup':
                parser.error("invalid target to enable: %r (not a target)" % (
                        name,))
            enable_targets.append(target)

    # Find the special library groups we are going to populate. We enforce that
    # these appear in the project (instead of just adding them) so that they at
    # least have an explicit representation in the project LLVMBuild files (and
    # comments explaining how they are populated).
    def find_special_group(name):
        info = info_map.get(name)
        if info is None:
            fatal("expected project to contain special %r component" % (
                    name,))

        if info.type_name != 'LibraryGroup':
            fatal("special component %r should be a LibraryGroup" % (
                    name,))

        if info.required_libraries:
            fatal("special component %r must have empty %r list" % (
                    name, 'required_libraries'))
        if info.add_to_library_groups:
            fatal("special component %r must have empty %r list" % (
                    name, 'add_to_library_groups'))

        info._is_special_group = True
        return info

    info_map = dict((ci.name, ci) for ci in project.component_infos)
    all_targets = find_special_group('all-targets')
    native_group = find_special_group('Native')
    native_codegen_group = find_special_group('NativeCodeGen')
    engine_group = find_special_group('Engine')

    # Set the enabled bit in all the target groups, and append to the
    # all-targets list.
    for ci in enable_targets:
        all_targets.required_libraries.append(ci.name)
        ci.enabled = True

    # If we have a native target, then that defines the native and
    # native_codegen libraries.
    if native_target and native_target.enabled:
        native_group.required_libraries.append(native_target.name)
        native_codegen_group.required_libraries.append(
            '%sCodeGen' % native_target.name)

    # If we have a native target with a JIT, use that for the engine. Otherwise,
    # use the interpreter.
    if native_target and native_target.enabled and native_target.has_jit:
        engine_group.required_libraries.append('JIT')
        engine_group.required_libraries.append(native_group.name)
    else:
        engine_group.required_libraries.append('Interpreter')

def main():
    from optparse import OptionParser, OptionGroup
    parser = OptionParser("usage: %prog [options]")

    group = OptionGroup(parser, "Input Options")
    group.add_option("", "--source-root", dest="source_root", metavar="PATH",
                      help="Path to the LLVM source (inferred if not given)",
                      action="store", default=None)
    group.add_option("", "--llvmbuild-source-root",
                     dest="llvmbuild_source_root",
                     help=(
            "If given, an alternate path to search for LLVMBuild.txt files"),
                     action="store", default=None, metavar="PATH")
    group.add_option("", "--build-root", dest="build_root", metavar="PATH",
                      help="Path to the build directory (if needed) [%default]",
                      action="store", default=None)
    parser.add_option_group(group)

    group = OptionGroup(parser, "Output Options")
    group.add_option("", "--print-tree", dest="print_tree",
                     help="Print out the project component tree [%default]",
                     action="store_true", default=False)
    group.add_option("", "--write-llvmbuild", dest="write_llvmbuild",
                      help="Write out the LLVMBuild.txt files to PATH",
                      action="store", default=None, metavar="PATH")
    group.add_option("", "--write-library-table",
                     dest="write_library_table", metavar="PATH",
                     help="Write the C++ library dependency table to PATH",
                     action="store", default=None)
    group.add_option("", "--write-cmake-fragment",
                     dest="write_cmake_fragment", metavar="PATH",
                     help="Write the CMake project information to PATH",
                     action="store", default=None)
    group.add_option("", "--write-make-fragment",
                      dest="write_make_fragment", metavar="PATH",
                     help="Write the Makefile project information to PATH",
                     action="store", default=None)
    group.add_option("", "--configure-target-def-file",
                     dest="configure_target_def_files",
                     help="""Configure the given file at SUBPATH (relative to
the inferred or given source root, and with a '.in' suffix) by replacing certain
substitution variables with lists of targets that support certain features (for
example, targets with AsmPrinters) and write the result to the build root (as
given by --build-root) at the same SUBPATH""",
                     metavar="SUBPATH", action="append", default=None)
    parser.add_option_group(group)

    group = OptionGroup(parser, "Configuration Options")
    group.add_option("", "--native-target",
                      dest="native_target", metavar="NAME",
                      help=("Treat the named target as the 'native' one, if "
                            "given [%default]"),
                      action="store", default=None)
    group.add_option("", "--enable-targets",
                      dest="enable_targets", metavar="NAMES",
                      help=("Enable the given space or semi-colon separated "
                            "list of targets, or all targets if not present"),
                      action="store", default=None)
    group.add_option("", "--enable-optional-components",
                      dest="optional_components", metavar="NAMES",
                      help=("Enable the given space or semi-colon separated "
                            "list of optional components"),
                      action="store", default=None)
    parser.add_option_group(group)

    (opts, args) = parser.parse_args()

    # Determine the LLVM source path, if not given.
    source_root = opts.source_root
    if source_root:
        if not os.path.exists(os.path.join(source_root, 'lib', 'VMCore',
                                           'Function.cpp')):
            parser.error('invalid LLVM source root: %r' % source_root)
    else:
        llvmbuild_path = os.path.dirname(__file__)
        llvm_build_path = os.path.dirname(llvmbuild_path)
        utils_path = os.path.dirname(llvm_build_path)
        source_root = os.path.dirname(utils_path)
        if not os.path.exists(os.path.join(source_root, 'lib', 'VMCore',
                                           'Function.cpp')):
            parser.error('unable to infer LLVM source root, please specify')

    # Construct the LLVM project information.
    llvmbuild_source_root = opts.llvmbuild_source_root or source_root
    project_info = LLVMProjectInfo.load_from_path(
        source_root, llvmbuild_source_root)

    # Add the magic target based components.
    add_magic_target_components(parser, project_info, opts)

    # Validate the project component info.
    project_info.validate_components()

    # Print the component tree, if requested.
    if opts.print_tree:
        project_info.print_tree()

    # Write out the components, if requested. This is useful for auto-upgrading
    # the schema.
    if opts.write_llvmbuild:
        project_info.write_components(opts.write_llvmbuild)

    # Write out the required library table, if requested.
    if opts.write_library_table:
        project_info.write_library_table(opts.write_library_table,
                                         opts.optional_components)

    # Write out the make fragment, if requested.
    if opts.write_make_fragment:
        project_info.write_make_fragment(opts.write_make_fragment)

    # Write out the cmake fragment, if requested.
    if opts.write_cmake_fragment:
        project_info.write_cmake_fragment(opts.write_cmake_fragment)

    # Configure target definition files, if requested.
    if opts.configure_target_def_files:
        # Verify we were given a build root.
        if not opts.build_root:
            parser.error("must specify --build-root when using "
                         "--configure-target-def-file")

        # Create the substitution list.
        available_targets = [ci for ci in project_info.component_infos
                             if ci.type_name == 'TargetGroup']
        substitutions = [
            ("@LLVM_ENUM_TARGETS@",
             ' '.join('LLVM_TARGET(%s)' % ci.name
                      for ci in available_targets)),
            ("@LLVM_ENUM_ASM_PRINTERS@",
             ' '.join('LLVM_ASM_PRINTER(%s)' % ci.name
                      for ci in available_targets
                      if ci.has_asmprinter)),
            ("@LLVM_ENUM_ASM_PARSERS@",
             ' '.join('LLVM_ASM_PARSER(%s)' % ci.name
                      for ci in available_targets
                      if ci.has_asmparser)),
            ("@LLVM_ENUM_DISASSEMBLERS@",
             ' '.join('LLVM_DISASSEMBLER(%s)' % ci.name
                      for ci in available_targets
                      if ci.has_disassembler))]

        # Configure the given files.
        for subpath in opts.configure_target_def_files:
            inpath = os.path.join(source_root, subpath + '.in')
            outpath = os.path.join(opts.build_root, subpath)
            result = configutil.configure_file(inpath, outpath, substitutions)
            if not result:
                note("configured file %r hasn't changed" % outpath)

if __name__=='__main__':
    main()
