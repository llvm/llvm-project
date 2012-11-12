; RUN: llc -filetype=obj -mtriple mipsel-unknown-linux %s -o - | elf-dump --dump-section-data  | FileCheck %s

; Check that the appropriate relocations were created.

; CHECK:     ('r_type', 0x2b)
; CHECK:     ('r_type', 0x2c)
; CHECK:     ('r_type', 0x2d)

@t1 = thread_local global i32 0, align 4

define i32 @f1() nounwind {
entry:
  %tmp = load i32* @t1, align 4
  ret i32 %tmp

}


@t2 = external thread_local global i32

define i32 @f2() nounwind {
entry:
  %tmp = load i32* @t2, align 4
  ret i32 %tmp

}

@f3.i = internal thread_local unnamed_addr global i32 1, align 4

define i32 @f3() nounwind {
entry:
  %0 = load i32* @f3.i, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, i32* @f3.i, align 4
  ret i32 %inc
}
