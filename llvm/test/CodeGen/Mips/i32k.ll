; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16a
; RUN: llc  -march=mipsel -mcpu=mips16 -relocation-model=pic -O3 < %s | FileCheck %s -check-prefix=16b

@.str = private unnamed_addr constant [4 x i8] c"%i\0A\00", align 1

define i32 @main() nounwind {
entry:
  %call = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i32 1075344593) nounwind
; 16a:	li	${{[0-9]+}}, 29905
; 16b:	li	${{[0-9]+}}, 16408
  %call1 = tail call i32 (i8*, ...)* @printf(i8* getelementptr inbounds ([4 x i8]* @.str, i32 0, i32 0), i32 -1075344593) nounwind
; 16a:	li	${{[0-9]+}}, 49127
; 16b:	li	${{[0-9]+}}, 35631
  ret i32 0
}

declare i32 @printf(i8* nocapture, ...) nounwind
