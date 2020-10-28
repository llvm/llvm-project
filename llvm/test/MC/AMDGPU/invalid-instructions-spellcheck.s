# RUN: not llvm-mc -triple amdgcn < %s 2>&1 | FileCheck --strict-whitespace %s

# This tests the mnemonic spell checker.

# First check what happens when an instruction is omitted:

v2, v4, v6

# CHECK:      error: invalid instruction
# CHECK-NEXT:{{^}}v2, v4, v6
# CHECK-NEXT:{{^}}^

# We don't want to see a suggestion here; the edit distance is too large to
# give sensible suggestions:

aaaaaaaaaaaaaaa v1, v2, v3

# CHECK:      error: invalid instruction
# CHECK-NEXT:{{^}}aaaaaaaaaaaaaaa v1, v2, v3
# CHECK-NEXT:{{^}}^

# Check that we get one suggestion: 'dsc_write_src2_b64' is 1 edit away, i.e. an deletion.

dsc_write_src2_b64 v1, v2, v3

# CHECK:      error: invalid instruction, did you mean: ds_write_src2_b64?
# CHECK-NEXT:{{^}}dsc_write_src2_b64 v1, v2, v3
# CHECK-NEXT:{{^}}^

# Check edit distance 1 and 2, just insertions:

s_mov_b v1, v2

# CHECK:      error: invalid instruction, did you mean: s_mov_b32, s_mov_b64?
# CHECK-NEXT:{{^}}s_mov_b v1, v2
# CHECK-NEXT:{{^}}^

# Check an instruction that is 2 edits away, and also has a lot of candidates:

s_load_dwordx v1, v2, v3

# CHECK:      error: invalid instruction, did you mean: s_load_dword, s_load_dwordx16, s_load_dwordx2, s_load_dwordx4, s_load_dwordx8?
# CHECK-NEXT:{{^}}s_load_dwordx v1, v2, v3
# CHECK-NEXT:{{^}}^
