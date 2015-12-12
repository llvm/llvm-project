; RUN: llc -mtriple=i686-pc-windows-msvc   < %s | FileCheck %s --check-prefix=X86
; RUN: llc -mtriple=x86_64-pc-windows-msvc < %s | FileCheck %s --check-prefix=X64

; Based on this source:
; extern "C" void may_throw(int);
; void f() {
;   try {
;     may_throw(1);
;     try {
;       may_throw(2);
;     } catch (int) {
;       may_throw(3);
;     }
;   } catch (int) {
;     may_throw(4);
;   }
; }

%rtti.TypeDescriptor2 = type { i8**, i8*, [3 x i8] }
%eh.CatchHandlerType = type { i32, i8* }

declare void @may_throw(i32)
declare i32 @__CxxFrameHandler3(...)
declare void @llvm.eh.begincatch(i8*, i8*)
declare void @llvm.eh.endcatch()
declare i32 @llvm.eh.typeid.for(i8*)

$"\01??_R0H@8" = comdat any

@"\01??_7type_info@@6B@" = external constant i8*
@"\01??_R0H@8" = linkonce_odr global %rtti.TypeDescriptor2 { i8** @"\01??_7type_info@@6B@", i8* null, [3 x i8] c".H\00" }, comdat
@llvm.eh.handlertype.H.0 = private unnamed_addr constant %eh.CatchHandlerType { i32 0, i8* bitcast (%rtti.TypeDescriptor2* @"\01??_R0H@8" to i8*) }, section "llvm.metadata"

define void @f() #0 personality i8* bitcast (i32 (...)* @__CxxFrameHandler3 to i8*) {
entry:
  invoke void @may_throw(i32 1)
          to label %invoke.cont unwind label %lpad.1

invoke.cont:                                      ; preds = %entry
  invoke void @may_throw(i32 2)
          to label %try.cont.9 unwind label %lpad

try.cont.9:                                       ; preds = %invoke.cont.3, %invoke.cont, %catch.7
  ret void

lpad:                                             ; preds = %catch, %entry
  %cs1 = catchswitch within none [label %catch] unwind label %lpad.1

catch:                                            ; preds = %lpad.1
  %p1 = catchpad within %cs1 [%rtti.TypeDescriptor2* @"\01??_R0H@8", i32 0, i8* null]
  invoke void @may_throw(i32 3)
          to label %invoke.cont.3 unwind label %lpad.1

invoke.cont.3:                                    ; preds = %catch
  catchret from %p1 to label %try.cont.9

lpad.1:                                           ; preds = %invoke.cont
  %cs2 = catchswitch within none [label %catch.7] unwind to caller

catch.7:
  %p2 = catchpad within %cs2 [%rtti.TypeDescriptor2* @"\01??_R0H@8", i32 0, i8* null]
  call void @may_throw(i32 4)
  catchret from %p2 to label %try.cont.9
}

; X86-LABEL: _f:
; X86: movl $-1, [[state:[-0-9]+]](%ebp)
; X86: movl $___ehhandler$f, {{.*}}
;
; X86: movl $0, [[state]](%ebp)
; X86: movl $1, (%esp)
; X86: calll _may_throw
;
; X86: movl $1, [[state]](%ebp)
; X86: movl $2, (%esp)
; X86: calll _may_throw
;
; X86: movl $2, [[state]](%ebp)
; X86: movl $3, (%esp)
; X86: calll _may_throw
;
; X86: movl $3, [[state]](%ebp)
; X86: movl $4, (%esp)
; X86: calll _may_throw


; X64-LABEL: f:
; X64-LABEL: $ip2state$f:
; X64-NEXT:   .long .Lfunc_begin0@IMGREL
; X64-NEXT:   .long -1
; X64-NEXT:   .long .Ltmp{{.*}}@IMGREL+1
; X64-NEXT:   .long 0
; X64-NEXT:   .long .Ltmp{{.*}}@IMGREL+1
; X64-NEXT:   .long 1
; X64-NEXT:   .long .Ltmp{{.*}}@IMGREL+1
; X64-NEXT:   .long -1
; X64-NEXT:   .long "?catch${{.*}}@?0?f@4HA"@IMGREL
; X64-NEXT:   .long 2
; X64-NEXT:   .long "?catch${{.*}}@?0?f@4HA"@IMGREL
; X64-NEXT:   .long 3

; Based on this source:
; extern "C" void may_throw(int);
; struct S { ~S(); };
; void g() {
;   S x;
;   try {
;     may_throw(-1);
;   } catch (...) {
;     may_throw(0);
;     {
;       S y;
;       may_throw(1);
;     }
;     may_throw(2);
;   }
; }

%struct.S = type { i8 }
declare void @"\01??1S@@QEAA@XZ"(%struct.S*)

define void @g() personality i32 (...)* @__CxxFrameHandler3 {
entry:
  %x = alloca %struct.S, align 1
  %y = alloca %struct.S, align 1
  invoke void @may_throw(i32 -1)
          to label %unreachable unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch] unwind label %ehcleanup5

catch:                                            ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null, i32 64, i8* null]
  invoke void @may_throw(i32 0)
          to label %invoke.cont unwind label %ehcleanup5

invoke.cont:                                      ; preds = %catch
  invoke void @may_throw(i32 1)
          to label %invoke.cont2 unwind label %ehcleanup

invoke.cont2:                                     ; preds = %invoke.cont
  invoke void @"\01??1S@@QEAA@XZ"(%struct.S* nonnull %y)
          to label %invoke.cont3 unwind label %ehcleanup5

invoke.cont3:                                     ; preds = %invoke.cont2
  invoke void @may_throw(i32 2)
          to label %invoke.cont4 unwind label %ehcleanup5

invoke.cont4:                                     ; preds = %invoke.cont3
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %invoke.cont4
  call void @"\01??1S@@QEAA@XZ"(%struct.S* nonnull %x)
  ret void

ehcleanup:                                        ; preds = %invoke.cont
  %2 = cleanuppad within %1 []
  call void @"\01??1S@@QEAA@XZ"(%struct.S* nonnull %y)
  cleanupret from %2 unwind label %ehcleanup5

ehcleanup5:                                       ; preds = %invoke.cont2, %invoke.cont3, %ehcleanup, %catch, %catch.dispatch
  %3 = cleanuppad within none []
  call void @"\01??1S@@QEAA@XZ"(%struct.S* nonnull %x)
  cleanupret from %3 unwind to caller

unreachable:                                      ; preds = %entry
  unreachable
}

; X86-LABEL: _g:
; X86: movl $-1, [[state:[-0-9]+]](%ebp)
; X86: movl $___ehhandler$g, {{.*}}
;
; X86: movl $1, [[state]](%ebp)
; X86: movl $-1, (%esp)
; X86: calll _may_throw
;
; X86: movl $2, [[state]](%ebp)
; X86: movl $0, (%esp)
; X86: calll _may_throw
;
; X86: movl $3, [[state]](%ebp)
; X86: movl $1, (%esp)
; X86: calll _may_throw
;
; X86: movl $2, [[state]](%ebp)
; X86: movl $2, (%esp)
; X86: calll _may_throw

; X64-LABEL: g:
; X64-LABEL: $ip2state$g:
; X64-NEXT:   .long .Lfunc_begin1@IMGREL
; X64-NEXT:   .long -1
; X64-NEXT:   .long .Ltmp{{.*}}@IMGREL+1
; X64-NEXT:   .long 1
; X64-NEXT:   .long .Ltmp{{.*}}@IMGREL+1
; X64-NEXT:   .long -1
; X64-NEXT:   .long "?catch${{.*}}@?0?g@4HA"@IMGREL
; X64-NEXT:   .long 2
; X64-NEXT:   .long .Ltmp{{.*}}@IMGREL+1
; X64-NEXT:   .long 3
; X64-NEXT:   .long .Ltmp{{.*}}@IMGREL+1
; X64-NEXT:   .long 2


; X86: .safeseh ___ehhandler$f
; X86: .safeseh ___ehhandler$g
