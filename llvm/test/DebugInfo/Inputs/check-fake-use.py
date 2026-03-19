#!/usr/bin/python3

# Parsing dwarfdump's output to determine whether the location list for the
# parameter "b" covers all of the function. The script searches for information
# in the input file to determine the [prologue, epilogue) range for the
# function, the location list range for "b", and checks that the latter covers
# the entirety of the former.
import re
import sys

DebugInfoPattern = r"\.debug_info contents:"
DebugLinePattern = r"\.debug_line contents:"
ProloguePattern = r"^\s*0x([0-9a-f]+)\s.+prologue_end"
EpiloguePattern = r"^\s*0x([0-9a-f]+)\s.+epilogue_begin"
FormalPattern = r"^0x[0-9a-f]+:\s+DW_TAG_formal_parameter"
LocationPattern = r"DW_AT_location\s+\[DW_FORM_([a-z_]+)\](?:.*0x([a-f0-9]+))"
DebugLocPattern = r'\[0x([a-f0-9]+),\s+0x([a-f0-9]+)\) ".text": (.+)$'

SeenDebugInfo = False
SeenDebugLine = False
LocationRanges = None
PrologueEnd = None
EpilogueBegin = None

# The dwarfdump output should contain the DW_AT_location for "b" first, then the
# line table which should contain prologue_end and epilogue_begin entries.
with open(sys.argv[1], "r") as dwarf_dump_file:
    dwarf_iter = iter(dwarf_dump_file)
    for line in dwarf_iter:
        if not SeenDebugInfo and re.match(DebugInfoPattern, line):
            SeenDebugInfo = True
        if not SeenDebugLine and re.match(DebugLinePattern, line):
            SeenDebugLine = True
        # Get the range of DW_AT_location for "b".
        if LocationRanges is None:
            if match := re.match(FormalPattern, line):
                # Go until we either find DW_AT_location or reach the end of this entry.
                location_match = None
                while location_match is None:
                    if (line := next(dwarf_iter, "")) == "\n":
                        raise RuntimeError(
                            ".debug_info output is missing DW_AT_location for 'b'"
                        )
                    location_match = re.search(LocationPattern, line)
                # Variable has whole-scope location, represented by an empty tuple.
                if location_match.group(1) == "exprloc":
                    LocationRanges = ()
                    continue
                if location_match.group(1) != "sec_offset":
                    raise RuntimeError(
                        f"Unhandled form for DW_AT_location: DW_FORM_{location_match.group(1)}"
                    )
                # Variable has location range list.
                if (
                    debug_loc_match := re.search(DebugLocPattern, next(dwarf_iter, ""))
                ) is None:
                    raise RuntimeError(f"Invalid location range list for 'b'")
                LocationRanges = (
                    int(debug_loc_match.group(1), 16),
                    int(debug_loc_match.group(2), 16),
                )
                while (
                    debug_loc_match := re.search(DebugLocPattern, next(dwarf_iter, ""))
                ) is not None:
                    match_loc_start = int(debug_loc_match.group(1), 16)
                    match_loc_end = int(debug_loc_match.group(2), 16)
                    match_expr = debug_loc_match.group(3)
                    if match_loc_start != LocationRanges[1]:
                        raise RuntimeError(
                            f"Location list for 'b' is discontinuous from [0x{LocationRanges[1]:x}, 0x{match_loc_start:x})"
                        )
                    if "stack_value" in match_expr:
                        raise RuntimeError(
                            f"Location list for 'b' contains a stack_value expression: {match_expr}"
                        )
                    LocationRanges = (LocationRanges[0], match_loc_end)
        # Get the prologue_end address.
        elif PrologueEnd is None:
            if match := re.match(ProloguePattern, line):
                PrologueEnd = int(match.group(1), 16)
        # Get the epilogue_begin address.
        elif EpilogueBegin is None:
            if match := re.match(EpiloguePattern, line):
                EpilogueBegin = int(match.group(1), 16)
                break

if not SeenDebugInfo:
    raise RuntimeError(".debug_info section not found.")
if not SeenDebugLine:
    raise RuntimeError(".debug_line section not found.")

if LocationRanges is None:
    raise RuntimeError(".debug_info output is missing parameter 'b'")
if PrologueEnd is None:
    raise RuntimeError(".debug_line output is missing prologue_end")
if EpilogueBegin is None:
    raise RuntimeError(".debug_line output is missing epilogue_begin")

if len(LocationRanges) == 2 and (
    LocationRanges[0] > PrologueEnd or LocationRanges[1] < EpilogueBegin
):
    raise RuntimeError(
        f"""Location list for 'b' does not cover the whole function:")
    Prologue to Epilogue = [0x{PrologueEnd:x}, 0x{EpilogueBegin:x})
    Location range = [0x{LocationRanges[0]:x}, 0x{LocationRanges[1]:x})
"""
    )
