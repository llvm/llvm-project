; RUN: opt < %s -argpromotion -S | FileCheck %s
; PR 32917

@b = common local_unnamed_addr global i32 0, align 4
@a = common local_unnamed_addr global i32 0, align 4

define i32 @fn2() local_unnamed_addr {
  %1 = load i32, i32* @b, align 4
  %2 = sext i32 %1 to i64
  %3 = inttoptr i64 %2 to i32*
  call fastcc void @fn1(i32* %3)
  ret i32 undef
}

define internal fastcc void @fn1(i32* nocapture readonly) unnamed_addr {
  %2 = getelementptr inbounds i32, i32* %0, i64 -1
  %3 = load i32, i32* %2, align 4
  store i32 %3, i32* @a, align 4
  ret void
}

; CHECK: getelementptr {{.*}} -1
; CHECK-NOT: getelementptr {{.*}} 4294967295
