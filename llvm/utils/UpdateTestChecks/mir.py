"""MIR test utility functions for UpdateTestChecks scripts."""

import re
import sys
from UpdateTestChecks import common
from UpdateTestChecks.common import (
    CHECK_RE,
    warn,
)

IR_FUNC_NAME_RE = re.compile(
    r"^\s*define\s+(?:internal\s+)?[^@]*@(?P<func>[A-Za-z0-9_.]+)\s*\("
)
IR_PREFIX_DATA_RE = re.compile(r"^ *(;|$)")
MIR_FUNC_NAME_RE = re.compile(r" *name: *(?P<func>[A-Za-z0-9_.-]+)")
MIR_BODY_BEGIN_RE = re.compile(r" *body: *\|")
MIR_BASIC_BLOCK_RE = re.compile(r" *bb\.[0-9]+.*:$")
MIR_PREFIX_DATA_RE = re.compile(r"^ *(;|bb.[0-9].*: *$|[a-z]+:( |$)|$)")

VREG_RE = re.compile(r"(%[0-9]+)(?:\.[a-z0-9_]+)?(?::[a-z0-9_]+)?(?:\([<>a-z0-9 ]+\))?")
MI_FLAGS_STR = (
    r"(frame-setup |frame-destroy |nnan |ninf |nsz |arcp |contract |afn "
    r"|reassoc |nuw |nsw |exact |nofpexcept |nomerge |unpredictable "
    r"|noconvergent |nneg |disjoint |nusw |samesign |inbounds )*"
)
VREG_DEF_FLAGS_STR = r"(?:dead |undef )*"

# Pattern to match the defined vregs and the opcode of an instruction that
# defines vregs. Opcodes starting with a lower-case 't' are allowed to match
# ARM's thumb instructions, like tADDi8 and t2ADDri.
VREG_DEF_RE = re.compile(
    r"^ *(?P<vregs>{2}{0}(?:, {2}{0})*) = "
    r"{1}(?P<opcode>[A-Zt][A-Za-z0-9_]+)".format(
        VREG_RE.pattern, MI_FLAGS_STR, VREG_DEF_FLAGS_STR
    )
)

MIR_FUNC_RE = re.compile(
    r"^---$"
    r"\n"
    r"^ *name: *(?P<func>[A-Za-z0-9_.-]+)$"
    r".*?"
    r"(?:^ *fixedStack: *(\[\])? *\n"
    r"(?P<fixedStack>.*?)\n?"
    r"^ *stack:"
    r".*?)?"
    r"^ *body: *\|\n"
    r"(?P<body>.*?)\n"
    r"^\.\.\.$",
    flags=(re.M | re.S),
)


def build_function_info_dictionary(
    test, raw_tool_output, triple, prefixes, func_dict, verbose
):
    for m in MIR_FUNC_RE.finditer(raw_tool_output):
        func = m.group("func")
        fixedStack = m.group("fixedStack")
        body = m.group("body")
        if verbose:
            print("Processing function: {}".format(func), file=sys.stderr)
            for l in body.splitlines():
                print("  {}".format(l), file=sys.stderr)

        # Vreg mangling
        mangled = []
        vreg_map = {}
        for func_line in body.splitlines(keepends=True):
            m = VREG_DEF_RE.match(func_line)
            if m:
                for vreg in VREG_RE.finditer(m.group("vregs")):
                    if vreg.group(1) in vreg_map:
                        name = vreg_map[vreg.group(1)]
                    else:
                        name = mangle_vreg(m.group("opcode"), vreg_map.values())
                        vreg_map[vreg.group(1)] = name
                    func_line = func_line.replace(
                        vreg.group(1), "[[{}:%[0-9]+]]".format(name), 1
                    )
            for number, name in vreg_map.items():
                func_line = re.sub(
                    r"{}\b".format(number), "[[{}]]".format(name), func_line
                )
            mangled.append(func_line)
        body = "".join(mangled)

        for prefix in prefixes:
            info = common.function_body(
                body, fixedStack, None, None, None, None, ginfo=None
            )
            if func in func_dict[prefix]:
                if (
                    not func_dict[prefix][func]
                    or func_dict[prefix][func].scrub != info.scrub
                    or func_dict[prefix][func].extrascrub != info.extrascrub
                ):
                    func_dict[prefix][func] = None
            else:
                func_dict[prefix][func] = info


def mangle_vreg(opcode, current_names):
    base = opcode
    # Simplify some common prefixes and suffixes
    if opcode.startswith("G_"):
        base = base[len("G_") :]
    if opcode.endswith("_PSEUDO"):
        base = base[: len("_PSEUDO")]
    # Shorten some common opcodes with long-ish names
    base = dict(
        IMPLICIT_DEF="DEF",
        GLOBAL_VALUE="GV",
        CONSTANT="C",
        FCONSTANT="C",
        MERGE_VALUES="MV",
        UNMERGE_VALUES="UV",
        INTRINSIC="INT",
        INTRINSIC_W_SIDE_EFFECTS="INT",
        INSERT_VECTOR_ELT="IVEC",
        EXTRACT_VECTOR_ELT="EVEC",
        SHUFFLE_VECTOR="SHUF",
    ).get(base, base)
    # Avoid ambiguity when opcodes end in numbers
    if len(base.rstrip("0123456789")) < len(base):
        base += "_"

    i = 0
    for name in current_names:
        if name.rstrip("0123456789") == base:
            i += 1
    if i:
        return "{}{}".format(base, i)
    return base


def find_mir_functions_with_one_bb(lines, verbose=False):
    result = []
    cur_func = None
    bbs = 0
    for line in lines:
        m = MIR_FUNC_NAME_RE.match(line)
        if m:
            if bbs == 1:
                result.append(cur_func)
            cur_func = m.group("func")
            bbs = 0
        m = MIR_BASIC_BLOCK_RE.match(line)
        if m:
            bbs += 1
    if bbs == 1:
        result.append(cur_func)
    return result


def add_mir_checks_for_function(
    test,
    output_lines,
    run_list,
    func_dict,
    func_name,
    single_bb,
    print_fixed_stack,
    first_check_is_next,
    at_the_function_name,
    check_indent=None,
):
    printed_prefixes = set()
    for run in run_list:
        for prefix in run[0]:
            if prefix in printed_prefixes:
                break
            # func_info can be empty if there was a prefix conflict.
            if not func_dict[prefix].get(func_name):
                continue
            if printed_prefixes:
                # Add some space between different check prefixes.
                indent = len(output_lines[-1]) - len(output_lines[-1].lstrip(" "))
                output_lines.append(" " * indent + ";")
            printed_prefixes.add(prefix)
            add_mir_check_lines(
                test,
                output_lines,
                prefix,
                ("@" if at_the_function_name else "") + func_name,
                single_bb,
                func_dict[prefix][func_name],
                print_fixed_stack,
                first_check_is_next,
                check_indent,
            )
            break
        else:
            warn(
                "Found conflicting asm for function: {}".format(func_name),
                test_file=test,
            )
    return output_lines


def add_mir_check_lines(
    test,
    output_lines,
    prefix,
    func_name,
    single_bb,
    func_info,
    print_fixed_stack,
    first_check_is_next,
    check_indent=None,
):
    func_body = str(func_info).splitlines()
    if single_bb:
        # Don't bother checking the basic block label for a single BB
        func_body.pop(0)

    if not func_body:
        warn(
            "Function has no instructions to check: {}".format(func_name),
            test_file=test,
        )
        return

    first_line = func_body[0]
    indent = len(first_line) - len(first_line.lstrip(" "))
    # A check comment, indented the appropriate amount
    if check_indent is not None:
        check = "{}; {}".format(check_indent, prefix)
    else:
        check = "{:>{}}; {}".format("", indent, prefix)

    output_lines.append("{}-LABEL: name: {}".format(check, func_name))

    if print_fixed_stack:
        output_lines.append("{}: fixedStack:".format(check))
        for stack_line in func_info.extrascrub.splitlines():
            filecheck_directive = check + "-NEXT"
            output_lines.append("{}: {}".format(filecheck_directive, stack_line))

    first_check = not first_check_is_next
    for func_line in func_body:
        if not func_line.strip():
            # The mir printer prints leading whitespace so we can't use CHECK-EMPTY:
            output_lines.append(check + "-NEXT: {{" + func_line + "$}}")
            continue
        filecheck_directive = check if first_check else check + "-NEXT"
        first_check = False
        check_line = "{}: {}".format(filecheck_directive, func_line[indent:]).rstrip()
        output_lines.append(check_line)


def should_add_mir_line_to_output(input_line, prefix_set):
    # Skip any check lines that we're handling as well as comments
    m = CHECK_RE.match(input_line)
    if (m and m.group(1) in prefix_set) or input_line.strip() == ";":
        return False
    return True


def add_mir_checks(
    input_lines,
    prefix_set,
    autogenerated_note,
    test,
    run_list,
    func_dict,
    print_fixed_stack,
    first_check_is_next,
    at_the_function_name,
):
    simple_functions = find_mir_functions_with_one_bb(input_lines)

    output_lines = []
    output_lines.append(autogenerated_note)

    func_name = None
    state = "toplevel"
    for input_line in input_lines:
        if input_line == autogenerated_note:
            continue

        if state == "toplevel":
            m = IR_FUNC_NAME_RE.match(input_line)
            if m:
                state = "ir function prefix"
                func_name = m.group("func")
            if input_line.rstrip("| \r\n") == "---":
                state = "document"
            output_lines.append(input_line)
        elif state == "document":
            m = MIR_FUNC_NAME_RE.match(input_line)
            if m:
                state = "mir function metadata"
                func_name = m.group("func")
            if input_line.strip() == "...":
                state = "toplevel"
                func_name = None
            if should_add_mir_line_to_output(input_line, prefix_set):
                output_lines.append(input_line)
        elif state == "mir function metadata":
            if should_add_mir_line_to_output(input_line, prefix_set):
                output_lines.append(input_line)
            m = MIR_BODY_BEGIN_RE.match(input_line)
            if m:
                if func_name in simple_functions:
                    # If there's only one block, put the checks inside it
                    state = "mir function prefix"
                    continue
                state = "mir function body"
                add_mir_checks_for_function(
                    test,
                    output_lines,
                    run_list,
                    func_dict,
                    func_name,
                    single_bb=False,
                    print_fixed_stack=print_fixed_stack,
                    first_check_is_next=first_check_is_next,
                    at_the_function_name=at_the_function_name,
                )
        elif state == "mir function prefix":
            m = MIR_PREFIX_DATA_RE.match(input_line)
            if not m:
                state = "mir function body"
                add_mir_checks_for_function(
                    test,
                    output_lines,
                    run_list,
                    func_dict,
                    func_name,
                    single_bb=True,
                    print_fixed_stack=print_fixed_stack,
                    first_check_is_next=first_check_is_next,
                    at_the_function_name=at_the_function_name,
                )

            if should_add_mir_line_to_output(input_line, prefix_set):
                output_lines.append(input_line)
        elif state == "mir function body":
            if input_line.strip() == "...":
                state = "toplevel"
                func_name = None
            if should_add_mir_line_to_output(input_line, prefix_set):
                output_lines.append(input_line)
        elif state == "ir function prefix":
            m = IR_PREFIX_DATA_RE.match(input_line)
            if not m:
                state = "ir function body"
                add_mir_checks_for_function(
                    test,
                    output_lines,
                    run_list,
                    func_dict,
                    func_name,
                    single_bb=False,
                    print_fixed_stack=print_fixed_stack,
                    first_check_is_next=first_check_is_next,
                    at_the_function_name=at_the_function_name,
                )

            if should_add_mir_line_to_output(input_line, prefix_set):
                output_lines.append(input_line)
        elif state == "ir function body":
            if input_line.strip() == "}":
                state = "toplevel"
                func_name = None
            if should_add_mir_line_to_output(input_line, prefix_set):
                output_lines.append(input_line)
    return output_lines
