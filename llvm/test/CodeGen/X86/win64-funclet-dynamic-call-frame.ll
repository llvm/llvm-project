; RUN: llc -mtriple=x86_64-pc-windows-msvc -O0 < %s | FileCheck %s
; RUN: llc -mtriple=x86_64-pc-windows-msvc -O2 < %s | FileCheck %s
; Funclets already reserve outgoing argument space in their prologues, even
; when the parent has a variable-sized stack allocation.
declare i32 @__CxxFrameHandler3(...)
declare void @use(ptr)
declare void @may_throw(i64, i64, i64, i64, i64, i64)
define void @dynamic_catch(i64 %n) personality ptr @__CxxFrameHandler3 {
entry:
  %a = alloca i8, i64 %n, align 32
  invoke void @use(ptr %a) to label %done unwind label %dispatch
dispatch:
  %switch = catchswitch within none [label %catch] unwind to caller
catch:
  %pad = catchpad within %switch [ptr null, i32 64, ptr null]
  call void @may_throw(i64 1, i64 2, i64 3, i64 4, i64 5, i64 6) [ "funclet"(token %pad) ]
  catchret from %pad to label %done
done:
  ret void
}
; CHECK-LABEL: "?catch$
; CHECK: .seh_endprologue
; CHECK-NOT: subq {{.*}}%rsp
; CHECK: callq may_throw
; CHECK-NOT: addq {{.*}}%rsp
; CHECK: .seh_startepilogue
