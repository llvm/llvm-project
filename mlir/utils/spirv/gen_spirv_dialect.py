#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Script for updating SPIR-V dialect by scraping information from SPIR-V
# HTML and JSON specs from the Internet.
#
# For example, to define the enum attribute for SPIR-V memory model:
#
# ./gen_spirv_dialect.py --base-td-path /path/to/SPIRVBase.td \
#                        --new-enum MemoryModel
#
# The 'operand_kinds' dict of spirv.core.grammar.json contains all supported
# SPIR-V enum classes.

import itertools
import math
import re
import requests
import textwrap
import yaml

SPIRV_HTML_SPEC_URL = (
    "https://www.khronos.org/registry/spir-v/specs/unified1/SPIRV.html"
)
SPIRV_JSON_SPEC_URL = "https://raw.githubusercontent.com/KhronosGroup/SPIRV-Headers/master/include/spirv/unified1/spirv.core.grammar.json"

SPIRV_CL_EXT_HTML_SPEC_URL = "https://www.khronos.org/registry/SPIR-V/specs/unified1/OpenCL.ExtendedInstructionSet.100.html"
SPIRV_CL_EXT_JSON_SPEC_URL = "https://raw.githubusercontent.com/KhronosGroup/SPIRV-Headers/master/include/spirv/unified1/extinst.opencl.std.100.grammar.json"

AUTOGEN_OP_DEF_SEPARATOR = "\n// -----\n\n"
AUTOGEN_ENUM_SECTION_MARKER = "enum section. Generated from SPIR-V spec; DO NOT MODIFY!"
AUTOGEN_OPCODE_SECTION_MARKER = (
    "opcode section. Generated from SPIR-V spec; DO NOT MODIFY!"
)


def get_spirv_doc_from_html_spec(url, settings):
    """Extracts instruction documentation from SPIR-V HTML spec.

    Returns:
      - A dict mapping from instruction opcode to documentation.
    """
    if url is None:
        url = SPIRV_HTML_SPEC_URL

    response = requests.get(url)
    spec = response.content

    from bs4 import BeautifulSoup

    spirv = BeautifulSoup(spec, "html.parser")

    doc = {}

    if settings.gen_cl_ops:
        section_anchor = spirv.find("h2", {"id": "_binary_form"})
        for section in section_anchor.parent.find_all("div", {"class": "sect2"}):
            for table in section.find_all("table"):
                inst_html = table.tbody.tr.td
                opname = inst_html.a["id"]
                # Ignore the first line, which is just the opname.
                doc[opname] = inst_html.text.split("\n", 1)[1].strip()
    else:
        section_anchor = spirv.find("h3", {"id": "_instructions_3"})
        for section in section_anchor.parent.find_all("div", {"class": "sect3"}):
            for table in section.find_all("table"):
                inst_html = table.tbody.tr.td.p
                opname = inst_html.a["id"]
                # Ignore the first line, which is just the opname.
                doc[opname] = inst_html.text.split("\n", 1)[1].strip()

    return doc


def get_spirv_grammar_from_json_spec(url):
    """Extracts operand kind and instruction grammar from SPIR-V JSON spec.

    Returns:
      - A list containing all operand kinds' grammar
      - A list containing all instructions' grammar
    """
    response = requests.get(SPIRV_JSON_SPEC_URL)
    spec = response.content

    import json

    spirv = json.loads(spec)

    if url is None:
        return spirv["operand_kinds"], spirv["instructions"]

    response_ext = requests.get(url)
    spec_ext = response_ext.content
    spirv_ext = json.loads(spec_ext)

    return spirv["operand_kinds"], spirv_ext["instructions"]


def split_list_into_sublists(items):
    """Split the list of items into multiple sublists.

    This is to make sure the string composed from each sublist won't exceed
    80 characters.

    Arguments:
      - items: a list of strings
    """
    chuncks = []
    chunk = []
    chunk_len = 0

    for item in items:
        chunk_len += len(item) + 2
        if chunk_len > 80:
            chuncks.append(chunk)
            chunk = []
            chunk_len = len(item) + 2
        chunk.append(item)

    if len(chunk) != 0:
        chuncks.append(chunk)

    return chuncks


def uniquify_enum_cases(lst):
    """Prunes duplicate enum cases from the list.

    Arguments:
     - lst: List whose elements are to be uniqued. Assumes each element is a
       (symbol, value) pair and elements already sorted according to value.

    Returns:
     - A list with all duplicates removed. The elements are sorted according to
       value and, for each value, uniqued according to symbol.
       original list,
     - A map from deduplicated cases to the uniqued case.
    """
    cases = lst
    uniqued_cases = []
    duplicated_cases = {}

    # First sort according to the value
    cases.sort(key=lambda x: x[1])

    # Then group them according to the value
    for _, groups in itertools.groupby(cases, key=lambda x: x[1]):
        # For each value, sort according to the enumerant symbol.
        sorted_group = sorted(groups, key=lambda x: x[0])
        # Keep the "smallest" case, which is typically the symbol without extension
        # suffix. But we have special cases that we want to fix.
        case = sorted_group[0]
        for i in range(1, len(sorted_group)):
            duplicated_cases[sorted_group[i][0]] = case[0]
        if case[0] == "HlslSemanticGOOGLE":
            assert len(sorted_group) == 2, "unexpected new variant for HlslSemantic"
            case = sorted_group[1]
            duplicated_cases[sorted_group[0][0]] = case[0]
        uniqued_cases.append(case)

    return uniqued_cases, duplicated_cases


def toposort(dag, sort_fn):
    """Topologically sorts the given dag.

    Arguments:
      - dag: a dict mapping from a node to its incoming nodes.
      - sort_fn: a function for sorting nodes in the same batch.

    Returns:
      A list containing topologically sorted nodes.
    """

    # Returns the next batch of nodes without incoming edges
    def get_next_batch(dag):
        while True:
            no_prev_nodes = set(node for node, prev in dag.items() if not prev)
            if not no_prev_nodes:
                break
            yield sorted(no_prev_nodes, key=sort_fn)
            dag = {
                node: (prev - no_prev_nodes)
                for node, prev in dag.items()
                if node not in no_prev_nodes
            }
        assert not dag, "found cyclic dependency"

    sorted_nodes = []
    for batch in get_next_batch(dag):
        sorted_nodes.extend(batch)

    return sorted_nodes


def toposort_capabilities(all_cases, capability_mapping):
    """Returns topologically sorted capability (symbol, value) pairs.

    Arguments:
      - all_cases: all capability cases (containing symbol, value, and implied
        capabilities).
      - capability_mapping: mapping from duplicated capability symbols to the
        canonicalized symbol chosen for SPIRVBase.td.

    Returns:
      A list containing topologically sorted capability (symbol, value) pairs.
    """
    dag = {}
    name_to_value = {}
    for case in all_cases:
        # Get the current capability.
        cur = case["enumerant"]
        name_to_value[cur] = case["value"]
        # Ignore duplicated symbols.
        if cur in capability_mapping:
            continue

        # Get capabilities implied by the current capability.
        prev = case.get("capabilities", [])
        uniqued_prev = set([capability_mapping.get(c, c) for c in prev])
        dag[cur] = uniqued_prev

    sorted_caps = toposort(dag, lambda x: name_to_value[x])
    # Attach the capability's value as the second component of the pair.
    return [(c, name_to_value[c]) for c in sorted_caps]


def get_capability_mapping(operand_kinds):
    """Returns the capability mapping from duplicated cases to canonicalized ones.

    Arguments:
      - operand_kinds: all operand kinds' grammar spec

    Returns:
      - A map mapping from duplicated capability symbols to the canonicalized
        symbol chosen for SPIRVBase.td.
    """
    # Find the operand kind for capability
    cap_kind = {}
    for kind in operand_kinds:
        if kind["kind"] == "Capability":
            cap_kind = kind

    kind_cases = [(case["enumerant"], case["value"]) for case in cap_kind["enumerants"]]
    _, capability_mapping = uniquify_enum_cases(kind_cases)

    return capability_mapping


def get_availability_spec(enum_case, capability_mapping, for_op, for_cap):
    """Returns the availability specification string for the given enum case.

    Arguments:
      - enum_case: the enum case to generate availability spec for. It may contain
        'version', 'lastVersion', 'extensions', or 'capabilities'.
      - capability_mapping: mapping from duplicated capability symbols to the
        canonicalized symbol chosen for SPIRVBase.td.
      - for_op: bool value indicating whether this is the availability spec for an
        op itself.
      - for_cap: bool value indicating whether this is the availability spec for
        capabilities themselves.

    Returns:
      - A `let availability = [...];` string if with availability spec or
        empty string if without availability spec
    """
    assert not (for_op and for_cap), "cannot set both for_op and for_cap"

    DEFAULT_MIN_VERSION = "MinVersion<SPIRV_V_1_0>"
    DEFAULT_MAX_VERSION = "MaxVersion<SPIRV_V_1_6>"
    DEFAULT_CAP = "Capability<[]>"
    DEFAULT_EXT = "Extension<[]>"

    min_version = enum_case.get("version", "")
    if min_version == "None":
        min_version = ""
    elif min_version:
        min_version = "MinVersion<SPIRV_V_{}>".format(min_version.replace(".", "_"))
    # TODO: delete this once ODS can support dialect-specific content
    # and we can use omission to mean no requirements.
    if for_op and not min_version:
        min_version = DEFAULT_MIN_VERSION

    max_version = enum_case.get("lastVersion", "")
    if max_version:
        max_version = "MaxVersion<SPIRV_V_{}>".format(max_version.replace(".", "_"))
    # TODO: delete this once ODS can support dialect-specific content
    # and we can use omission to mean no requirements.
    if for_op and not max_version:
        max_version = DEFAULT_MAX_VERSION

    exts = enum_case.get("extensions", [])
    if exts:
        exts = "Extension<[{}]>".format(", ".join(sorted(set(exts))))
        # We need to strip the minimal version requirement if this symbol is
        # available via an extension, which means *any* SPIR-V version can support
        # it as long as the extension is provided. The grammar's 'version' field
        # under such case should be interpreted as this symbol is introduced as
        # a core symbol since the given version, rather than a minimal version
        # requirement.
        min_version = DEFAULT_MIN_VERSION if for_op else ""
    # TODO: delete this once ODS can support dialect-specific content
    # and we can use omission to mean no requirements.
    if for_op and not exts:
        exts = DEFAULT_EXT

    caps = enum_case.get("capabilities", [])
    implies = ""
    if caps:
        canonicalized_caps = []
        for c in caps:
            if c in capability_mapping:
                canonicalized_caps.append(capability_mapping[c])
            else:
                canonicalized_caps.append(c)
        prefixed_caps = [
            "SPIRV_C_{}".format(c) for c in sorted(set(canonicalized_caps))
        ]
        if for_cap:
            # If this is generating the availability for capabilities, we need to
            # put the capability "requirements" in implies field because now
            # the "capabilities" field in the source grammar means so.
            caps = ""
            implies = "list<I32EnumAttrCase> implies = [{}];".format(
                ", ".join(prefixed_caps)
            )
        else:
            caps = "Capability<[{}]>".format(", ".join(prefixed_caps))
            implies = ""
    # TODO: delete this once ODS can support dialect-specific content
    # and we can use omission to mean no requirements.
    if for_op and not caps:
        caps = DEFAULT_CAP

    avail = ""
    # Compose availability spec if any of the requirements is not empty.
    # For ops, because we have a default in SPIRV_Op class, omit if the spec
    # is the same.
    if (min_version or max_version or caps or exts) and not (
        for_op
        and min_version == DEFAULT_MIN_VERSION
        and max_version == DEFAULT_MAX_VERSION
        and caps == DEFAULT_CAP
        and exts == DEFAULT_EXT
    ):
        joined_spec = ",\n    ".join(
            [e for e in [min_version, max_version, exts, caps] if e]
        )
        avail = "{} availability = [\n    {}\n  ];".format(
            "let" if for_op else "list<Availability>", joined_spec
        )

    return "{}{}{}".format(implies, "\n  " if implies and avail else "", avail)


def gen_operand_kind_enum_attr(operand_kind, capability_mapping):
    """Generates the TableGen EnumAttr definition for the given operand kind.

    Returns:
      - The operand kind's name
      - A string containing the TableGen EnumAttr definition
    """
    if "enumerants" not in operand_kind:
        return "", ""

    # Returns a symbol for the given case in the given kind. This function
    # handles Dim specially to avoid having numbers as the start of symbols,
    # which does not play well with C++ and the MLIR parser.
    def get_case_symbol(kind_name, case_name):
        if kind_name == "Dim":
            if case_name == "1D" or case_name == "2D" or case_name == "3D":
                return "Dim{}".format(case_name)
        return case_name

    kind_name = operand_kind["kind"]
    is_bit_enum = operand_kind["category"] == "BitEnum"
    kind_acronym = "".join([c for c in kind_name if c >= "A" and c <= "Z"])

    name_to_case_dict = {}
    for case in operand_kind["enumerants"]:
        name_to_case_dict[case["enumerant"]] = case

    if kind_name == "Capability":
        # Special treatment for capability cases: we need to sort them topologically
        # because a capability can refer to another via the 'implies' field.
        kind_cases = toposort_capabilities(
            operand_kind["enumerants"], capability_mapping
        )
    else:
        kind_cases = [
            (case["enumerant"], case["value"]) for case in operand_kind["enumerants"]
        ]
        kind_cases, _ = uniquify_enum_cases(kind_cases)
    max_len = max([len(symbol) for (symbol, _) in kind_cases])

    # Generate the definition for each enum case
    case_category = "I32Bit" if is_bit_enum else "I32"
    fmt_str = (
        "def SPIRV_{acronym}_{case_name} {colon:>{offset}} "
        '{category}EnumAttrCase{suffix}<"{symbol}"{case_value_part}>{avail}'
    )
    case_defs = []
    for case_pair in kind_cases:
        name = case_pair[0]
        if is_bit_enum:
            value = int(case_pair[1], base=16)
        else:
            value = int(case_pair[1])
        avail = get_availability_spec(
            name_to_case_dict[name],
            capability_mapping,
            False,
            kind_name == "Capability",
        )
        if is_bit_enum:
            if value == 0:
                suffix = "None"
                value = ""
            else:
                suffix = "Bit"
                value = ", {}".format(int(math.log2(value)))
        else:
            suffix = ""
            value = ", {}".format(value)

        case_def = fmt_str.format(
            category=case_category,
            suffix=suffix,
            acronym=kind_acronym,
            case_name=name,
            symbol=get_case_symbol(kind_name, name),
            case_value_part=value,
            avail=" {{\n  {}\n}}".format(avail) if avail else ";",
            colon=":",
            offset=(max_len + 1 - len(name)),
        )
        case_defs.append(case_def)
    case_defs = "\n".join(case_defs)

    # Generate the list of enum case names
    fmt_str = "SPIRV_{acronym}_{symbol}"
    case_names = [
        fmt_str.format(acronym=kind_acronym, symbol=case[0]) for case in kind_cases
    ]

    # Split them into sublists and concatenate into multiple lines
    case_names = split_list_into_sublists(case_names)
    case_names = ["{:6}".format("") + ", ".join(sublist) for sublist in case_names]
    case_names = ",\n".join(case_names)

    # Generate the enum attribute definition
    kind_category = "Bit" if is_bit_enum else "I32"
    enum_attr = """def SPIRV_{name}Attr :
    SPIRV_{category}EnumAttr<"{name}", "valid SPIR-V {name}", "{snake_name}", [
{cases}
    ]>;""".format(
        name=kind_name,
        snake_name=snake_casify(kind_name),
        category=kind_category,
        cases=case_names,
    )
    return kind_name, case_defs + "\n\n" + enum_attr


def gen_opcode(instructions):
    """Generates the TableGen definition to map opname to opcode

    Returns:
      - A string containing the TableGen SPIRV_OpCode definition
    """

    max_len = max([len(inst["opname"]) for inst in instructions])
    def_fmt_str = (
        "def SPIRV_OC_{name} {colon:>{offset}} " 'I32EnumAttrCase<"{name}", {value}>;'
    )
    opcode_defs = [
        def_fmt_str.format(
            name=inst["opname"],
            value=inst["opcode"],
            colon=":",
            offset=(max_len + 1 - len(inst["opname"])),
        )
        for inst in instructions
    ]
    opcode_str = "\n".join(opcode_defs)

    decl_fmt_str = "SPIRV_OC_{name}"
    opcode_list = [decl_fmt_str.format(name=inst["opname"]) for inst in instructions]
    opcode_list = split_list_into_sublists(opcode_list)
    opcode_list = ["{:6}".format("") + ", ".join(sublist) for sublist in opcode_list]
    opcode_list = ",\n".join(opcode_list)
    enum_attr = (
        "def SPIRV_OpcodeAttr :\n"
        '    SPIRV_I32EnumAttr<"{name}", "valid SPIR-V instructions", '
        '"opcode", [\n'
        "{lst}\n"
        "    ]>;".format(name="Opcode", lst=opcode_list)
    )
    return opcode_str + "\n\n" + enum_attr


def map_cap_to_opnames(instructions):
    """Maps capabilities to instructions enabled by those capabilities

    Arguments:
      - instructions: a list containing a subset of SPIR-V instructions' grammar
    Returns:
      - A map with keys representing capabilities and values of lists of
      instructions enabled by the corresponding key
    """
    cap_to_inst = {}

    for inst in instructions:
        caps = inst["capabilities"] if "capabilities" in inst else ["0_core_0"]
        for cap in caps:
            if cap not in cap_to_inst:
                cap_to_inst[cap] = []
            cap_to_inst[cap].append(inst["opname"])

    return cap_to_inst


def gen_instr_coverage_report(path, instructions):
    """Dumps to standard output a YAML report of current instruction coverage

    Arguments:
      - path: the path to SPIRBase.td
      - instructions: a list containing all SPIR-V instructions' grammar
    """
    with open(path, "r") as f:
        content = f.read()

    content = content.split(AUTOGEN_OPCODE_SECTION_MARKER)

    prefix = "def SPIRV_OC_"
    existing_opcodes = [
        k[len(prefix) :] for k in re.findall(prefix + "\w+", content[1])
    ]
    existing_instructions = list(
        filter(lambda inst: (inst["opname"] in existing_opcodes), instructions)
    )

    instructions_opnames = [inst["opname"] for inst in instructions]

    remaining_opcodes = list(set(instructions_opnames) - set(existing_opcodes))
    remaining_instructions = list(
        filter(lambda inst: (inst["opname"] in remaining_opcodes), instructions)
    )

    rem_cap_to_instr = map_cap_to_opnames(remaining_instructions)
    ex_cap_to_instr = map_cap_to_opnames(existing_instructions)

    rem_cap_to_cov = {}

    # Calculate coverage for each capability
    for cap in rem_cap_to_instr:
        if cap not in ex_cap_to_instr:
            rem_cap_to_cov[cap] = 0.0
        else:
            rem_cap_to_cov[cap] = len(ex_cap_to_instr[cap]) / (
                len(ex_cap_to_instr[cap]) + len(rem_cap_to_instr[cap])
            )

    report = {}

    # Merge the 3 maps into one report
    for cap in rem_cap_to_instr:
        report[cap] = {}
        report[cap]["Supported Instructions"] = (
            ex_cap_to_instr[cap] if cap in ex_cap_to_instr else []
        )
        report[cap]["Unsupported Instructions"] = rem_cap_to_instr[cap]
        report[cap]["Coverage"] = "{}%".format(int(rem_cap_to_cov[cap] * 100))

    print(yaml.dump(report))


def update_td_opcodes(path, instructions, filter_list):
    """Updates SPIRBase.td with new generated opcode cases.

    Arguments:
      - path: the path to SPIRBase.td
      - instructions: a list containing all SPIR-V instructions' grammar
      - filter_list: a list containing new opnames to add
    """

    with open(path, "r") as f:
        content = f.read()

    content = content.split(AUTOGEN_OPCODE_SECTION_MARKER)
    assert len(content) == 3

    # Extend opcode list with existing list
    prefix = "def SPIRV_OC_"
    existing_opcodes = [
        k[len(prefix) :] for k in re.findall(prefix + "\w+", content[1])
    ]
    filter_list.extend(existing_opcodes)
    filter_list = list(set(filter_list))

    # Generate the opcode for all instructions in SPIR-V
    filter_instrs = list(
        filter(lambda inst: (inst["opname"] in filter_list), instructions)
    )
    # Sort instruction based on opcode
    filter_instrs.sort(key=lambda inst: inst["opcode"])
    opcode = gen_opcode(filter_instrs)

    # Substitute the opcode
    content = (
        content[0]
        + AUTOGEN_OPCODE_SECTION_MARKER
        + "\n\n"
        + opcode
        + "\n\n// End "
        + AUTOGEN_OPCODE_SECTION_MARKER
        + content[2]
    )

    with open(path, "w") as f:
        f.write(content)


def update_td_enum_attrs(path, operand_kinds, filter_list):
    """Updates SPIRBase.td with new generated enum definitions.

    Arguments:
      - path: the path to SPIRBase.td
      - operand_kinds: a list containing all operand kinds' grammar
      - filter_list: a list containing new enums to add
    """
    with open(path, "r") as f:
        content = f.read()

    content = content.split(AUTOGEN_ENUM_SECTION_MARKER)
    assert len(content) == 3

    # Extend filter list with existing enum definitions
    prefix = "def SPIRV_"
    suffix = "Attr"
    existing_kinds = [
        k[len(prefix) : -len(suffix)]
        for k in re.findall(prefix + "\w+" + suffix, content[1])
    ]
    filter_list.extend(existing_kinds)

    capability_mapping = get_capability_mapping(operand_kinds)

    # Generate definitions for all enums in filter list
    defs = [
        gen_operand_kind_enum_attr(kind, capability_mapping)
        for kind in operand_kinds
        if kind["kind"] in filter_list
    ]
    # Sort alphabetically according to enum name
    defs.sort(key=lambda enum: enum[0])
    # Only keep the definitions from now on
    # Put Capability's definition at the very beginning because capability cases
    # will be referenced later
    defs = [enum[1] for enum in defs if enum[0] == "Capability"] + [
        enum[1] for enum in defs if enum[0] != "Capability"
    ]

    # Substitute the old section
    content = (
        content[0]
        + AUTOGEN_ENUM_SECTION_MARKER
        + "\n\n"
        + "\n\n".join(defs)
        + "\n\n// End "
        + AUTOGEN_ENUM_SECTION_MARKER
        + content[2]
    )

    with open(path, "w") as f:
        f.write(content)


def snake_casify(name):
    """Turns the given name to follow snake_case convention."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def map_spec_operand_to_ods_argument(operand):
    """Maps an operand in SPIR-V JSON spec to an op argument in ODS.

    Arguments:
      - A dict containing the operand's kind, quantifier, and name

    Returns:
      - A string containing both the type and name for the argument
    """
    kind = operand["kind"]
    quantifier = operand.get("quantifier", "")

    # These instruction "operands" are for encoding the results; they should
    # not be handled here.
    assert kind != "IdResultType", 'unexpected to handle "IdResultType" kind'
    assert kind != "IdResult", 'unexpected to handle "IdResult" kind'

    if kind == "IdRef":
        if quantifier == "":
            arg_type = "SPIRV_Type"
        elif quantifier == "?":
            arg_type = "Optional<SPIRV_Type>"
        else:
            arg_type = "Variadic<SPIRV_Type>"
    elif kind == "IdMemorySemantics" or kind == "IdScope":
        # TODO: Need to further constrain 'IdMemorySemantics'
        # and 'IdScope' given that they should be generated from OpConstant.
        assert quantifier == "", (
            "unexpected to have optional/variadic memory " "semantics or scope <id>"
        )
        arg_type = "SPIRV_" + kind[2:] + "Attr"
    elif kind == "LiteralInteger":
        if quantifier == "":
            arg_type = "I32Attr"
        elif quantifier == "?":
            arg_type = "OptionalAttr<I32Attr>"
        else:
            arg_type = "OptionalAttr<I32ArrayAttr>"
    elif (
        kind == "LiteralString"
        or kind == "LiteralContextDependentNumber"
        or kind == "LiteralExtInstInteger"
        or kind == "LiteralSpecConstantOpInteger"
        or kind == "PairLiteralIntegerIdRef"
        or kind == "PairIdRefLiteralInteger"
        or kind == "PairIdRefIdRef"
    ):
        assert False, '"{}" kind unimplemented'.format(kind)
    else:
        # The rest are all enum operands that we represent with op attributes.
        assert quantifier != "*", "unexpected to have variadic enum attribute"
        arg_type = "SPIRV_{}Attr".format(kind)
        if quantifier == "?":
            arg_type = "OptionalAttr<{}>".format(arg_type)

    name = operand.get("name", "")
    name = snake_casify(name) if name else kind.lower()

    return "{}:${}".format(arg_type, name)


def get_description(text, appendix):
    """Generates the description for the given SPIR-V instruction.

    Arguments:
      - text: Textual description of the operation as string.
      - appendix: Additional contents to attach in description as string,
                  includking IR examples, and others.

    Returns:
      - A string that corresponds to the description of the Tablegen op.
    """
    fmt_str = "{text}\n\n    <!-- End of AutoGen section -->\n{appendix}\n  "
    return fmt_str.format(text=text, appendix=appendix)


def get_op_definition(
    instruction, opname, doc, existing_info, capability_mapping, settings
):
    """Generates the TableGen op definition for the given SPIR-V instruction.

    Arguments:
      - instruction: the instruction's SPIR-V JSON grammar
      - doc: the instruction's SPIR-V HTML doc
      - existing_info: a dict containing potential manually specified sections for
        this instruction
      - capability_mapping: mapping from duplicated capability symbols to the
                     canonicalized symbol chosen for SPIRVBase.td

    Returns:
      - A string containing the TableGen op definition
    """
    if settings.gen_cl_ops:
        fmt_str = (
            "def SPIRV_{opname}Op : "
            'SPIRV_{inst_category}<"{opname_src}", {opcode}, <<Insert result type>> > '
            "{{\n  let summary = {summary};\n\n  let description = "
            "[{{\n{description}}}];{availability}\n"
        )
    else:
        fmt_str = (
            "def SPIRV_{vendor_name}{opname_src}Op : "
            'SPIRV_{inst_category}<"{opname_src}"{category_args}, [{traits}]> '
            "{{\n  let summary = {summary};\n\n  let description = "
            "[{{\n{description}}}];{availability}\n"
        )

    vendor_name = ""
    inst_category = existing_info.get("inst_category", "Op")
    if inst_category == "Op":
        fmt_str += (
            "\n  let arguments = (ins{args});\n\n" "  let results = (outs{results});\n"
        )
    elif inst_category.endswith("VendorOp"):
        vendor_name = inst_category.split("VendorOp")[0].upper()
        assert len(vendor_name) != 0, "Invalid instruction category"

    fmt_str += "{extras}" "}}\n"

    opname_src = instruction["opname"]
    if opname.startswith("Op"):
        opname_src = opname_src[2:]
    if len(vendor_name) > 0:
        assert opname_src.endswith(
            vendor_name
        ), "op name does not match the instruction category"
        opname_src = opname_src[: -len(vendor_name)]

    category_args = existing_info.get("category_args", "")

    if "\n" in doc:
        summary, text = doc.split("\n", 1)
    else:
        summary = doc
        text = ""
    wrapper = textwrap.TextWrapper(
        width=76, initial_indent="    ", subsequent_indent="    "
    )

    # Format summary. If the summary can fit in the same line, we print it out
    # as a "-quoted string; otherwise, wrap the lines using "[{...}]".
    summary = summary.strip()
    if len(summary) + len('  let summary = "";') <= 80:
        summary = '"{}"'.format(summary)
    else:
        summary = "[{{\n{}\n  }}]".format(wrapper.fill(summary))

    # Wrap text
    text = text.split("\n")
    text = [wrapper.fill(line) for line in text if line]
    text = "\n\n".join(text)

    operands = instruction.get("operands", [])

    # Op availability
    avail = get_availability_spec(instruction, capability_mapping, True, False)
    if avail:
        avail = "\n\n  {0}".format(avail)

    # Set op's result
    results = ""
    if len(operands) > 0 and operands[0]["kind"] == "IdResultType":
        results = "\n    SPIRV_Type:$result\n  "
        operands = operands[1:]
    if "results" in existing_info:
        results = existing_info["results"]

    # Ignore the operand standing for the result <id>
    if len(operands) > 0 and operands[0]["kind"] == "IdResult":
        operands = operands[1:]

    # Set op' argument
    arguments = existing_info.get("arguments", None)
    if arguments is None:
        arguments = [map_spec_operand_to_ods_argument(o) for o in operands]
        arguments = ",\n    ".join(arguments)
        if arguments:
            # Prepend and append whitespace for formatting
            arguments = "\n    {}\n  ".format(arguments)

    description = existing_info.get("description", None)
    if description is None:
        assembly = (
            "\n    ```\n"
            "    [TODO]\n"
            "    ```\n\n"
            "    #### Example:\n\n"
            "    ```mlir\n"
            "    [TODO]\n"
            "    ```"
        )
        description = get_description(text, assembly)

    return fmt_str.format(
        opname=opname,
        opname_src=opname_src,
        opcode=instruction["opcode"],
        category_args=category_args,
        inst_category=inst_category,
        vendor_name=vendor_name,
        traits=existing_info.get("traits", ""),
        summary=summary,
        description=description,
        availability=avail,
        args=arguments,
        results=results,
        extras=existing_info.get("extras", ""),
    )


def get_string_between(base, start, end):
    """Extracts a substring with a specified start and end from a string.

    Arguments:
      - base: string to extract from.
      - start: string to use as the start of the substring.
      - end: string to use as the end of the substring.

    Returns:
      - The substring if found
      - The part of the base after end of the substring. Is the base string itself
        if the substring wasnt found.
    """
    split = base.split(start, 1)
    if len(split) == 2:
        rest = split[1].split(end, 1)
        assert len(rest) == 2, (
            'cannot find end "{end}" while extracting substring '
            "starting with {start}".format(start=start, end=end)
        )
        return rest[0].rstrip(end), rest[1]
    return "", split[0]


def get_string_between_nested(base, start, end):
    """Extracts a substring with a nested start and end from a string.

    Arguments:
      - base: string to extract from.
      - start: string to use as the start of the substring.
      - end: string to use as the end of the substring.

    Returns:
      - The substring if found
      - The part of the base after end of the substring. Is the base string itself
        if the substring wasn't found.
    """
    split = base.split(start, 1)
    if len(split) == 2:
        # Handle nesting delimiters
        rest = split[1]
        unmatched_start = 1
        index = 0
        while unmatched_start > 0 and index < len(rest):
            if rest[index:].startswith(end):
                unmatched_start -= 1
                if unmatched_start == 0:
                    break
                index += len(end)
            elif rest[index:].startswith(start):
                unmatched_start += 1
                index += len(start)
            else:
                index += 1

        assert index < len(rest), (
            'cannot find end "{end}" while extracting substring '
            'starting with "{start}"'.format(start=start, end=end)
        )
        return rest[:index], rest[index + len(end) :]
    return "", split[0]


def extract_td_op_info(op_def):
    """Extracts potentially manually specified sections in op's definition.

    Arguments: - A string containing the op's TableGen definition

    Returns:
      - A dict containing potential manually specified sections
    """
    # Get opname
    prefix = "def SPIRV_"
    suffix = "Op"
    opname = [
        o[len(prefix) : -len(suffix)]
        for o in re.findall(prefix + "\w+" + suffix, op_def)
    ]
    assert len(opname) == 1, "more than one ops in the same section!"
    opname = opname[0]

    # Get instruction category
    prefix = "SPIRV_"
    inst_category = [
        o[len(prefix) :] for o in re.findall(prefix + "\w+Op", op_def.split(":", 1)[1])
    ]
    assert len(inst_category) <= 1, "more than one ops in the same section!"
    inst_category = inst_category[0] if len(inst_category) == 1 else "Op"

    # Get category_args
    op_tmpl_params, _ = get_string_between_nested(op_def, "<", ">")
    opstringname, rest = get_string_between(op_tmpl_params, '"', '"')
    category_args = rest.split("[", 1)[0]

    # Get traits
    traits, _ = get_string_between_nested(rest, "[", "]")

    # Get description
    description, rest = get_string_between(op_def, "let description = [{\n", "}];\n")

    # Get arguments
    args, rest = get_string_between(rest, "  let arguments = (ins", ");\n")

    # Get results
    results, rest = get_string_between(rest, "  let results = (outs", ");\n")

    extras = rest.strip(" }\n")
    if extras:
        extras = "\n  {}\n".format(extras)

    return {
        # Prefix with 'Op' to make it consistent with SPIR-V spec
        "opname": "Op{}".format(opname),
        "inst_category": inst_category,
        "category_args": category_args,
        "traits": traits,
        "description": description,
        "arguments": args,
        "results": results,
        "extras": extras,
    }


def update_td_op_definitions(
    path, instructions, docs, filter_list, inst_category, capability_mapping, settings
):
    """Updates SPIRVOps.td with newly generated op definition.

    Arguments:
      - path: path to SPIRVOps.td
      - instructions: SPIR-V JSON grammar for all instructions
      - docs: SPIR-V HTML doc for all instructions
      - filter_list: a list containing new opnames to include
      - capability_mapping: mapping from duplicated capability symbols to the
                     canonicalized symbol chosen for SPIRVBase.td.

    Returns:
      - A string containing all the TableGen op definitions
    """
    with open(path, "r") as f:
        content = f.read()

    # Split the file into chunks, each containing one op.
    ops = content.split(AUTOGEN_OP_DEF_SEPARATOR)
    header = ops[0]
    footer = ops[-1]
    ops = ops[1:-1]

    # For each existing op, extract the manually-written sections out to retain
    # them when re-generating the ops. Also append the existing ops to filter
    # list.
    name_op_map = {}  # Map from opname to its existing ODS definition
    op_info_dict = {}
    for op in ops:
        info_dict = extract_td_op_info(op)
        opname = info_dict["opname"]
        name_op_map[opname] = op
        op_info_dict[opname] = info_dict
        filter_list.append(opname)
    filter_list = sorted(list(set(filter_list)))

    op_defs = []

    if settings.gen_cl_ops:
        fix_opname = lambda src: src.replace("CL", "").lower()
    else:
        fix_opname = lambda src: src

    for opname in filter_list:
        # Find the grammar spec for this op
        try:
            fixed_opname = fix_opname(opname)
            instruction = next(
                inst for inst in instructions if inst["opname"] == fixed_opname
            )

            op_defs.append(
                get_op_definition(
                    instruction,
                    opname,
                    docs[fixed_opname],
                    op_info_dict.get(opname, {"inst_category": inst_category}),
                    capability_mapping,
                    settings,
                )
            )
        except StopIteration:
            # This is an op added by us; use the existing ODS definition.
            op_defs.append(name_op_map[opname])

    # Substitute the old op definitions
    op_defs = [header] + op_defs + [footer]
    content = AUTOGEN_OP_DEF_SEPARATOR.join(op_defs)

    with open(path, "w") as f:
        f.write(content)


if __name__ == "__main__":
    import argparse

    cli_parser = argparse.ArgumentParser(
        description="Update SPIR-V dialect definitions using SPIR-V spec"
    )

    cli_parser.add_argument(
        "--base-td-path",
        dest="base_td_path",
        type=str,
        default=None,
        help="Path to SPIRVBase.td",
    )
    cli_parser.add_argument(
        "--op-td-path",
        dest="op_td_path",
        type=str,
        default=None,
        help="Path to SPIRVOps.td",
    )

    cli_parser.add_argument(
        "--new-enum",
        dest="new_enum",
        type=str,
        default=None,
        help="SPIR-V enum to be added to SPIRVBase.td",
    )
    cli_parser.add_argument(
        "--new-opcodes",
        dest="new_opcodes",
        type=str,
        default=None,
        nargs="*",
        help="update SPIR-V opcodes in SPIRVBase.td",
    )
    cli_parser.add_argument(
        "--new-inst",
        dest="new_inst",
        type=str,
        default=None,
        nargs="*",
        help="SPIR-V instruction to be added to ops file",
    )
    cli_parser.add_argument(
        "--inst-category",
        dest="inst_category",
        type=str,
        default="Op",
        help="SPIR-V instruction category used for choosing "
        "the TableGen base class to define this op",
    )
    cli_parser.add_argument(
        "--gen-cl-ops",
        dest="gen_cl_ops",
        help="Generate OpenCL Extended Instruction Set op",
        action="store_true",
    )
    cli_parser.set_defaults(gen_cl_ops=False)
    cli_parser.add_argument(
        "--gen-inst-coverage", dest="gen_inst_coverage", action="store_true"
    )
    cli_parser.set_defaults(gen_inst_coverage=False)

    args = cli_parser.parse_args()

    if args.gen_cl_ops:
        ext_html_url = SPIRV_CL_EXT_HTML_SPEC_URL
        ext_json_url = SPIRV_CL_EXT_JSON_SPEC_URL
    else:
        ext_html_url = None
        ext_json_url = None

    operand_kinds, instructions = get_spirv_grammar_from_json_spec(ext_json_url)

    # Define new enum attr
    if args.new_enum is not None:
        assert args.base_td_path is not None
        filter_list = [args.new_enum] if args.new_enum else []
        update_td_enum_attrs(args.base_td_path, operand_kinds, filter_list)

    # Define new opcode
    if args.new_opcodes is not None:
        assert args.base_td_path is not None
        update_td_opcodes(args.base_td_path, instructions, args.new_opcodes)

    # Define new op
    if args.new_inst is not None:
        assert args.op_td_path is not None
        docs = get_spirv_doc_from_html_spec(ext_html_url, args)
        capability_mapping = get_capability_mapping(operand_kinds)
        update_td_op_definitions(
            args.op_td_path,
            instructions,
            docs,
            args.new_inst,
            args.inst_category,
            capability_mapping,
            args,
        )
        print("Done. Note that this script just generates a template; ", end="")
        print("please read the spec and update traits, arguments, and ", end="")
        print("results accordingly.")

    if args.gen_inst_coverage:
        gen_instr_coverage_report(args.base_td_path, instructions)
