@ RUN: not llvm-mc -triple=arm -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:
@ RUN: not llvm-mc -triple=arm-apple-darwin -filetype=obj %s -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=error:

	.align 3
symbol:
@ CHECK:      :[[#@LINE+1]]:6: error: unsupported relocation type
.quad(symbol)
@ CHECK:      :[[#@LINE+1]]:8: error: unsupported relocation type
.8byte symbol
