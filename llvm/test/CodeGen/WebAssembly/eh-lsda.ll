; RUN: llc < %s --mtriple=wasm32-unknown-unknown -wasm-disable-explicit-locals -wasm-keep-registers -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling | FileCheck %s -check-prefixes=CHECK,NOPIC -DPTR=32
; RUN: llc < %s --mtriple=wasm64-unknown-unknown -wasm-disable-explicit-locals -wasm-keep-registers -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling | FileCheck %s -check-prefixes=CHECK,NOPIC -DPTR=64
; RUN: llc < %s --mtriple=wasm32-unknown-emscripten -wasm-disable-explicit-locals -wasm-keep-registers -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling | FileCheck %s -check-prefixes=CHECK,NOPIC -DPTR=32
; RUN: llc < %s --mtriple=wasm64-unknown-emscripten -wasm-disable-explicit-locals -wasm-keep-registers -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling | FileCheck %s -check-prefixes=CHECK,NOPIC -DPTR=64
; RUN: llc < %s --mtriple=wasm32-unknown-emscripten -wasm-disable-explicit-locals -wasm-keep-registers -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling -relocation-model=pic | FileCheck %s -check-prefixes=CHECK,PIC -DPTR=32
; RUN: llc < %s --mtriple=wasm64-unknown-emscripten -wasm-disable-explicit-locals -wasm-keep-registers -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling -relocation-model=pic | FileCheck %s -check-prefixes=CHECK,PIC -DPTR=64

@_ZTIi = external constant ptr
@_ZTIf = external constant ptr
@_ZTId = external constant ptr

; Single catch (...) does not need an exception table.
;
; try {
;   may_throw();
; } catch (...) {
; }
; CHECK-LABEL: test0:
; CHECK-NOT: GCC_except_table
define void @test0() personality ptr @__gxx_wasm_personality_v0 {
entry:
  invoke void @may_throw()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr null]
  %2 = call ptr @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call ptr @__cxa_begin_catch(ptr %2) [ "funclet"(token %1) ]
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %entry, %catch.start
  ret void
}

; Exception table generation + shared action test.
;
; try {
;   may_throw();
; } catch (int) {
; } catch (float) {
; } catch (double) {
; } catch (...) {
; }
;
; try {
;   may_throw();
; } catch (double) {
; } catch (...) {
; }
;
; try {
;   may_throw();
; } catch (int) {
; } catch (float) {
; }
;
; There are three landing pads. The second landing pad should share action table
; entries with the first landing pad because they end with the same sequence
; (double -> ...). But the third landing table cannot share action table entries
; with others, so it should create its own entries.

; CHECK-LABEL: test1:
; In static linking, we load GCC_except_table as a constant directly.
; NOPIC:      i[[PTR]].const  $push[[CONTEXT:.*]]=, {{[48]}}
; NOPIC-NEXT: i[[PTR]].const  $push[[EXCEPT_TABLE:.*]]=, GCC_except_table1
; NOPIC-NEXT: i[[PTR]].store  __wasm_lpad_context($pop[[CONTEXT]]), $pop[[EXCEPT_TABLE]]

; In case of PIC, we make GCC_except_table symbols a relative on based on
; __memory_base.
; PIC:        global.get  $push[[CONTEXT:.*]]=, __wasm_lpad_context@GOT
; PIC-NEXT:   local.tee  $push{{.*}}=, $[[CONTEXT_LOCAL:.*]]=, $pop[[CONTEXT]]
; PIC:        global.get  $push[[MEMORY_BASE:.*]]=, __memory_base
; PIC-NEXT:   i[[PTR]].const  $push[[EXCEPT_TABLE_REL:.*]]=, GCC_except_table1@MBREL
; PIC-NEXT:   i[[PTR]].add   $push[[EXCEPT_TABLE:.*]]=, $pop[[MEMORY_BASE]], $pop[[EXCEPT_TABLE_REL]]
; PIC-NEXT:   i[[PTR]].store  {{[48]}}($[[CONTEXT_LOCAL]]), $pop[[EXCEPT_TABLE]]

; CHECK: .section  .rodata.gcc_except_table,"",@
; CHECK-NEXT:   .p2align  2
; CHECK-NEXT: GCC_except_table[[START:[0-9]+]]:
; CHECK-NEXT: .Lexception0:
; CHECK-NEXT:   .int8  255                     # @LPStart Encoding = omit
; CHECK-NEXT:   .int8  0                       # @TType Encoding = absptr
; CHECK-NEXT:   .uleb128 .Lttbase0-.Lttbaseref0
; CHECK-NEXT: .Lttbaseref0:
; CHECK-NEXT:   .int8  1                       # Call site Encoding = uleb128
; CHECK-NEXT:   .uleb128 .Lcst_end0-.Lcst_begin0
; CHECK-NEXT: .Lcst_begin0:
; CHECK-NEXT:   .int8  0                       # >> Call Site 0 <<
; CHECK-NEXT:                                  #   On exception at call site 0
; CHECK-NEXT:   .int8  7                       #   Action: 4
; CHECK-NEXT:   .int8  1                       # >> Call Site 1 <<
; CHECK-NEXT:                                  #   On exception at call site 1
; CHECK-NEXT:   .int8  3                       #   Action: 2
; CHECK-NEXT:   .int8  2                       # >> Call Site 2 <<
; CHECK-NEXT:                                  #   On exception at call site 2
; CHECK-NEXT:   .int8  11                      #   Action: 6
; CHECK-NEXT: .Lcst_end0:
; CHECK-NEXT:   .int8  1                       # >> Action Record 1 <<
; CHECK-NEXT:                                  #   Catch TypeInfo 1
; CHECK-NEXT:   .int8  0                       #   No further actions
; CHECK-NEXT:   .int8  2                       # >> Action Record 2 <<
; CHECK-NEXT:                                  #   Catch TypeInfo 2
; CHECK-NEXT:   .int8  125                     #   Continue to action 1
; CHECK-NEXT:   .int8  3                       # >> Action Record 3 <<
; CHECK-NEXT:                                  #   Catch TypeInfo 3
; CHECK-NEXT:   .int8  125                     #   Continue to action 2
; CHECK-NEXT:   .int8  4                       # >> Action Record 4 <<
; CHECK-NEXT:                                  #   Catch TypeInfo 4
; CHECK-NEXT:   .int8  125                     #   Continue to action 3
; CHECK-NEXT:   .int8  3                       # >> Action Record 5 <<
; CHECK-NEXT:                                  #   Catch TypeInfo 3
; CHECK-NEXT:   .int8  0                       #   No further actions
; CHECK-NEXT:   .int8  4                       # >> Action Record 6 <<
; CHECK-NEXT:                                  #   Catch TypeInfo 4
; CHECK-NEXT:   .int8  125                     #   Continue to action 5
; CHECK-NEXT:   .p2align  2
; CHECK-NEXT:                                  # >> Catch TypeInfos <<
; CHECK-NEXT:   .int[[PTR]]  _ZTIi             # TypeInfo 4
; CHECK-NEXT:   .int[[PTR]]  _ZTIf             # TypeInfo 3
; CHECK-NEXT:   .int[[PTR]]  _ZTId             # TypeInfo 2
; CHECK-NEXT:   .int[[PTR]]  0                 # TypeInfo 1
; CHECK-NEXT: .Lttbase0:
; CHECK-NEXT:   .p2align  2
; CHECK-NEXT: .LGCC_except_table_end[[END:[0-9]+]]:
; CHECK-NEXT:   .size  GCC_except_table[[START]], .LGCC_except_table_end[[END]]-GCC_except_table[[START]]
define void @test1() personality ptr @__gxx_wasm_personality_v0 {
entry:
  invoke void @may_throw()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [ptr @_ZTIi, ptr @_ZTIf, ptr @_ZTId, ptr null]
  %2 = call ptr @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i32 @llvm.eh.typeid.for(ptr @_ZTIi)
  %matches = icmp eq i32 %3, %4
  br i1 %matches, label %catch10, label %catch.fallthrough

catch10:                                          ; preds = %catch.start
  %5 = call ptr @__cxa_begin_catch(ptr %2) [ "funclet"(token %1) ]
  %6 = load i32, ptr %5, align 4
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %entry, %catch, %catch4, %catch7, %catch10
  invoke void @may_throw()
          to label %try.cont23 unwind label %catch.dispatch14

catch.dispatch14:                                 ; preds = %try.cont
  %7 = catchswitch within none [label %catch.start15] unwind to caller

catch.start15:                                    ; preds = %catch.dispatch14
  %8 = catchpad within %7 [ptr @_ZTId, ptr null]
  %9 = call ptr @llvm.wasm.get.exception(token %8)
  %10 = call i32 @llvm.wasm.get.ehselector(token %8)
  %11 = call i32 @llvm.eh.typeid.for(ptr @_ZTId)
  %matches16 = icmp eq i32 %10, %11
  %12 = call ptr @__cxa_begin_catch(ptr %9) [ "funclet"(token %8) ]
  br i1 %matches16, label %catch20, label %catch17

catch20:                                          ; preds = %catch.start15
  %13 = load double, ptr %12, align 8
  call void @__cxa_end_catch() [ "funclet"(token %8) ]
  catchret from %8 to label %try.cont23

try.cont23:                                       ; preds = %try.cont, %catch17, %catch20
  invoke void @may_throw()
          to label %try.cont36 unwind label %catch.dispatch25

catch.dispatch25:                                 ; preds = %try.cont23
  %14 = catchswitch within none [label %catch.start26] unwind to caller

catch.start26:                                    ; preds = %catch.dispatch25
  %15 = catchpad within %14 [ptr @_ZTIi, ptr @_ZTIf]
  %16 = call ptr @llvm.wasm.get.exception(token %15)
  %17 = call i32 @llvm.wasm.get.ehselector(token %15)
  %18 = call i32 @llvm.eh.typeid.for(ptr @_ZTIi)
  %matches27 = icmp eq i32 %17, %18
  br i1 %matches27, label %catch33, label %catch.fallthrough28

catch33:                                          ; preds = %catch.start26
  %19 = call ptr @__cxa_begin_catch(ptr %16) [ "funclet"(token %15) ]
  %20 = load i32, ptr %19, align 4
  call void @__cxa_end_catch() [ "funclet"(token %15) ]
  catchret from %15 to label %try.cont36

catch.fallthrough28:                              ; preds = %catch.start26
  %21 = call i32 @llvm.eh.typeid.for(ptr @_ZTIf)
  %matches29 = icmp eq i32 %17, %21
  br i1 %matches29, label %catch30, label %rethrow

catch30:                                          ; preds = %catch.fallthrough28
  %22 = call ptr @__cxa_begin_catch(ptr %16) [ "funclet"(token %15) ]
  %23 = load float, ptr %22, align 4
  call void @__cxa_end_catch() [ "funclet"(token %15) ]
  catchret from %15 to label %try.cont36

rethrow:                                          ; preds = %catch.fallthrough28
  call void @__cxa_rethrow() [ "funclet"(token %15) ]
  unreachable

try.cont36:                                       ; preds = %try.cont23, %catch30, %catch33
  ret void

catch17:                                          ; preds = %catch.start15
  call void @__cxa_end_catch() [ "funclet"(token %8) ]
  catchret from %8 to label %try.cont23

catch.fallthrough:                                ; preds = %catch.start
  %24 = call i32 @llvm.eh.typeid.for(ptr @_ZTIf)
  %matches1 = icmp eq i32 %3, %24
  br i1 %matches1, label %catch7, label %catch.fallthrough2

catch7:                                           ; preds = %catch.fallthrough
  %25 = call ptr @__cxa_begin_catch(ptr %2) [ "funclet"(token %1) ]
  %26 = load float, ptr %25, align 4
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

catch.fallthrough2:                               ; preds = %catch.fallthrough
  %27 = call i32 @llvm.eh.typeid.for(ptr @_ZTId)
  %matches3 = icmp eq i32 %3, %27
  %28 = call ptr @__cxa_begin_catch(ptr %2) [ "funclet"(token %1) ]
  br i1 %matches3, label %catch4, label %catch

catch4:                                           ; preds = %catch.fallthrough2
  %29 = load double, ptr %28, align 8
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

catch:                                            ; preds = %catch.fallthrough2
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont
}

declare void @may_throw()
; Function Attrs: nounwind
declare i32 @llvm.eh.typeid.for(ptr) #0
; Function Attrs: nounwind
declare ptr @llvm.wasm.get.exception(token) #0
; Function Attrs: nounwind
declare i32 @llvm.wasm.get.ehselector(token) #0
declare void @__cxa_rethrow()
declare ptr @__cxa_begin_catch(ptr)
declare void @__cxa_end_catch()
declare i32 @__gxx_wasm_personality_v0(...)

attributes #0 = { nounwind }
