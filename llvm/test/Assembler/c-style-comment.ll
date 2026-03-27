; RUN: llvm-as < %s | llvm-dis | FileCheck %s

/* Simple C style comment */

; CHECK: @B = external global i32
@B = external global i32

/* multiline C ctyle comment at "top-level"
 * This is the second line
 * and this is third
 */


; CHECK: @foo
define <4 x i1> @foo(<4 x float> %a, <4 x float> %b) nounwind {
entry: /* inline comment */
  %cmp = fcmp olt <4 x float> %a, /* to be ignored */ %b
  ret <4 x i1> %cmp /* ignore */
}

/* End of the assembly file */

