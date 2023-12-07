; RUN: sed -e s/.Cxx:// %s | llc -mtriple=x86_64-pc-windows-msvc | FileCheck %s --check-prefix=CXX
; RUN: sed -e s/.Seh:// %s | llc -mtriple=x86_64-pc-windows-msvc | FileCheck %s --check-prefix=SEH

declare i32 @__CxxFrameHandler3(...)
declare i32 @__C_specific_handler(...)
declare void @dummy_filter()

declare void @f(i32)

;Cxx: define void @test() personality ptr @__CxxFrameHandler3 {
;Seh: define void @test() personality ptr @__C_specific_handler {
entry:
  invoke void @f(i32 1)
          to label %invoke.cont unwind label %catch.dispatch

catch.dispatch:
  %cs1 = catchswitch within none [label %catch.body] unwind label %catch.dispatch.2

catch.body:
;Cxx: %catch = catchpad within %cs1 [ptr null, i32 u0x40, ptr null]
;Seh: %catch = catchpad within %cs1 [ptr @dummy_filter]
  invoke void @f(i32 2) [ "funclet"(token %catch) ]
          to label %unreachable unwind label %terminate

terminate:
  %cleanup = cleanuppad within %catch []
  call void @f(i32 3) [ "funclet"(token %cleanup) ]
  unreachable

unreachable:
  unreachable

invoke.cont:
  ret void

catch.dispatch.2:
  %cs2 = catchswitch within none [label %catch.body.2] unwind to caller

catch.body.2:
;Cxx: %catch2 = catchpad within %cs2 [ptr null, i32 u0x40, ptr null]
;Seh: %catch2 = catchpad within %cs2 [ptr @dummy_filter]
  unreachable
}

; CXX-LABEL: test:
; CXX-LABEL: $ip2state$test:
; CXX-NEXT:   .long   .Lfunc_begin0@IMGREL
; CXX-NEXT:   .long   -1
; CXX-NEXT:   .long   .Ltmp0@IMGREL+1
; CXX-NEXT:   .long   1
; CXX-NEXT:   .long   .Ltmp1@IMGREL+1
; CXX-NEXT:   .long   -1
; CXX-NEXT:   .long   "?catch$3@?0?test@4HA"@IMGREL
; CXX-NEXT:   .long   2
; CXX-NEXT:   .long   .Ltmp2@IMGREL+1
; CXX-NEXT:   .long   3
; CXX-NEXT:   .long   .Ltmp3@IMGREL+1
; CXX-NEXT:   .long   2
; CXX-NEXT:   .long   "?catch$5@?0?test@4HA"@IMGREL
; CXX-NEXT:   .long   4

; SEH-LABEL: test:
; SEH-LABEL: .Llsda_begin0:
; SEH-NEXT:    .long   .Ltmp0@IMGREL
; SEH-NEXT:    .long   .Ltmp1@IMGREL+1
; SEH-NEXT:    .long   dummy_filter@IMGREL
; SEH-NEXT:    .long   .LBB0_3@IMGREL
; SEH-NEXT:    .long   .Ltmp0@IMGREL
; SEH-NEXT:    .long   .Ltmp1@IMGREL+1
; SEH-NEXT:    .long   dummy_filter@IMGREL
; SEH-NEXT:    .long   .LBB0_5@IMGREL
; SEH-NEXT:    .long   .Ltmp2@IMGREL
; SEH-NEXT:    .long   .Ltmp3@IMGREL+1
; SEH-NEXT:    .long   "?dtor$2@?0?test@4HA"@IMGREL
; SEH-NEXT:    .long   0
; SEH-NEXT:    .long   .Ltmp2@IMGREL
; SEH-NEXT:    .long   .Ltmp3@IMGREL+1
; SEH-NEXT:    .long   dummy_filter@IMGREL
; SEH-NEXT:    .long   .LBB0_5@IMGREL
; SEH-NEXT:  .Llsda_end0:
