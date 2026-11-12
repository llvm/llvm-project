; RUN: opt -passes=simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S < %s | FileCheck %s
target datalayout = "e-m:x-p:32:32-i64:64-f80:32-n8:16:32-a:0:32-S32"
target triple = "i686-pc-win32"

declare i32 @__CxxFrameHandler3(...)

declare void @fn_2()

define void @fn_1(i1 %B) personality ptr @__CxxFrameHandler3 {
entry:
  br i1 %B, label %__Ea.exit, label %lor.lhs.false.i.i

lor.lhs.false.i.i:
  br i1 %B, label %if.end.i.i, label %__Ea.exit

if.end.i.i:
  invoke void @fn_2()
          to label %__Ea.exit unwind label %ehcleanup.i

ehcleanup.i:
  %t4 = cleanuppad within none []
  br label %arraydestroy.body.i

arraydestroy.body.i:
  %gep = getelementptr i8, ptr null, i32 -1
  br label %dtor.exit.i

dtor.exit.i:
  br i1 %B, label %arraydestroy.done3.i, label %arraydestroy.body.i

arraydestroy.done3.i:
  cleanupret from %t4 unwind to caller

__Ea.exit:
  ret void
}

; CHECK-LABEL: define void @fn_1(
; CHECK-NEXT: entry:
; CHECK-NEXT: ret void
