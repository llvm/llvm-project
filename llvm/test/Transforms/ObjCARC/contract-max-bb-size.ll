; RUN: opt -objc-arc-contract -S < %s | FileCheck -check-prefix=ENABLE %s
; RUN: opt -objc-arc-contract -arc-contract-max-bb-size=3 -S < %s | FileCheck -check-prefix=DISABLE %s

@g0 = common global i8* null, align 8

; ENABLE: store i8* %2, i8** @g0
; DISABLE: store i8* %1, i8** @g0

define void @foo0() {
  %1 = tail call i8* @foo1()
  %2 = tail call i8* @llvm.objc.retainAutoreleasedReturnValue(i8* %1)
  store i8* %1, i8** @g0, align 8
  ret void
}

declare i8* @foo1()
declare i8* @llvm.objc.retainAutoreleasedReturnValue(i8*)
