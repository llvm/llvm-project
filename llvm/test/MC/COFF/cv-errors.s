# RUN: not llvm-mc -triple=i686-pc-win32 %s -o /dev/null 2>&1 | FileCheck %s

.text
foo:
.cv_file a
# CHECK: error: expected file number in '.cv_file' directive
# CHECK-NOT: error:
.cv_file 0 "t.cpp"
# CHECK: error: file number less than one
# CHECK-NOT: error:
.cv_func_id x
# CHECK: error: expected function id in '.cv_func_id' directive
# CHECK-NOT: error:
.cv_func_id -1
# CHECK: error: expected function id in '.cv_func_id' directive
# CHECK-NOT: error:
.cv_func_id 0xFFFFFFFFFFFFFFFF
# CHECK: error: expected function id within range [0, UINT_MAX)
# CHECK-NOT: error:
.cv_inline_site_id x
# CHECK: error: expected function id in '.cv_inline_site_id' directive
# CHECK-NOT: error:

.cv_file 1 "t.cpp"
.cv_func_id 0

.cv_inline_site_id 0 0 0 0 0 0
# CHECK: error: expected 'within' identifier in '.cv_inline_site_id' directive
# CHECK-NOT: error:

.cv_inline_site_id 0 within a
# CHECK: error: expected function id in '.cv_inline_site_id' directive
# CHECK-NOT: error:

.cv_inline_site_id 0 within 0 x
# CHECK: error: expected 'inlined_at' identifier in '.cv_inline_site_id' directive
# CHECK-NOT: error:

.cv_inline_site_id 0 within 0 inlined_at 0 0 0
# CHECK: error: file number less than one in '.cv_inline_site_id' directive
# CHECK-NOT: error:

.cv_inline_site_id 0 within 0 inlined_at 10 0 0
# CHECK: error: unassigned file number in '.cv_inline_site_id' directive
# CHECK-NOT: error:

.cv_inline_site_id 0 within 0 inlined_at 1 1 1
# CHECK: error: function id already allocated
# CHECK-NOT: error:

.cv_inline_site_id 1 within 0 inlined_at 1 1 1

.cv_loc 0 1 1 1 # t.cpp:1:1
nop
.cv_loc 1 1 1 1 # t.cpp:1:1
nop
