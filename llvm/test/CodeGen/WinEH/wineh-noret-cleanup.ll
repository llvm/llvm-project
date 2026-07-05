; RUN: sed -e s/.Cxx:// %s | llc -mtriple=x86_64-pc-windows-msvc | FileCheck %s --check-prefixes=CXX
; RUN: sed -e s/.Seh:// %s | llc -mtriple=x86_64-pc-windows-msvc | FileCheck %s --check-prefixes=SEH
; RUN: %if aarch64-registered-target %{ sed -e s/.Cxx:// %s | llc -mtriple=aarch64-pc-windows-msvc | FileCheck %s --check-prefix=CXX %}
; RUN: %if aarch64-registered-target %{ sed -e s/.Seh:// %s | llc -mtriple=aarch64-pc-windows-msvc | FileCheck %s --check-prefix=SEH %}

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
; CXX-NEXT:   .[[ENTRY:long|word]]   .Lfunc_begin0@IMGREL
; CXX-NEXT:   .[[ENTRY]]   -1
; CXX-NEXT:   .[[ENTRY]]   .Ltmp0@IMGREL
; CXX-NEXT:   .[[ENTRY]]   1
; CXX-NEXT:   .[[ENTRY]]   .Ltmp1@IMGREL
; CXX-NEXT:   .[[ENTRY]]   -1
; CXX-NEXT:   .[[ENTRY]]   "?catch$3@?0?test@4HA"@IMGREL
; CXX-NEXT:   .[[ENTRY]]   2
; CXX-NEXT:   .[[ENTRY]]   .Ltmp2@IMGREL
; CXX-NEXT:   .[[ENTRY]]   3
; CXX-NEXT:   .[[ENTRY]]   .Ltmp3@IMGREL
; CXX-NEXT:   .[[ENTRY]]   2
; CXX-NEXT:   .[[ENTRY]]   "?catch$5@?0?test@4HA"@IMGREL
; CXX-NEXT:   .[[ENTRY]]   4

; SEH-LABEL: test:
; SEH:        .LBB0_[[CATCH2:[0-9]+]]: {{.*}} %catch.body.2
; SEH:        .LBB0_[[CATCH:[0-9]+]]: {{.*}} %catch.body
; SEH-LABEL: .Llsda_begin0:
; SEH-NEXT:    .[[ENTRY:long|word]]   .Ltmp0@IMGREL
; SEH-NEXT:    .[[ENTRY]]   .Ltmp1@IMGREL
; SEH-NEXT:    .[[ENTRY]]   dummy_filter@IMGREL
; SEH-NEXT:    .[[ENTRY]]   .LBB0_[[CATCH]]@IMGREL
; SEH-NEXT:    .[[ENTRY]]   .Ltmp0@IMGREL
; SEH-NEXT:    .[[ENTRY]]   .Ltmp1@IMGREL
; SEH-NEXT:    .[[ENTRY]]   dummy_filter@IMGREL
; SEH-NEXT:    .[[ENTRY]]   .LBB0_[[CATCH2]]@IMGREL
; SEH-NEXT:    .[[ENTRY]]   .Ltmp2@IMGREL
; SEH-NEXT:    .[[ENTRY]]   .Ltmp3@IMGREL
; SEH-NEXT:    .[[ENTRY]]   "?dtor$[[DTOR:[0-9]+]]@?0?test@4HA"@IMGREL
; SEH-NEXT:    .[[ENTRY]]   0
; SEH-NEXT:    .[[ENTRY]]   .Ltmp2@IMGREL
; SEH-NEXT:    .[[ENTRY]]   .Ltmp3@IMGREL
; SEH-NEXT:    .[[ENTRY]]   dummy_filter@IMGREL
; SEH-NEXT:    .[[ENTRY]]   .LBB0_[[CATCH2]]@IMGREL
; SEH-NEXT:  .Llsda_end0:
; SEH:        "?dtor$[[DTOR]]@?0?test@4HA"
