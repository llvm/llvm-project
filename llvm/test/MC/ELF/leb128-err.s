# RUN: not llvm-mc -filetype=obj -triple=x86_64 %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

.section .alloc_w,"aw",@progbits; w:
# CHECK: :[[#@LINE+1]]:16: error: .uleb128 expression is not absolute
.uleb128 extern-w   # extern is undefined
# CHECK: :[[#@LINE+1]]:11: error: .uleb128 expression is not absolute
.uleb128 w-extern
# CHECK: :[[#@LINE+1]]:11: error: .uleb128 expression is not absolute
.uleb128 x-w        # x is later defined in another section
.uleb128 w1-w       # w1 is later defined in the same section
w1:

.section .alloc_x,"aw",@progbits; x:
# CHECK: :[[#@LINE+1]]:11: error: .sleb128 expression is not absolute
.sleb128 y-x
.section .alloc_y,"aw",@progbits; y:
# CHECK: :[[#@LINE+1]]:11: error: .sleb128 expression is not absolute
.sleb128 x-y

.section .nonalloc_x; nx:
# CHECK: :[[#@LINE+1]]:12: error: .sleb128 expression is not absolute
.sleb128 ny-nx
.section .nonalloc_y; ny:
# CHECK: :[[#@LINE+1]]:12: error: .sleb128 expression is not absolute
.sleb128 nx-ny
