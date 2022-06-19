; RUN: llc < %s --mtriple=wasm32-unknown-unknown -wasm-disable-explicit-locals -wasm-keep-registers -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling | FileCheck %s -check-prefixes=CHECK,NOPIC -DPTR=32
; RUN: llc < %s --mtriple=wasm64-unknown-unknown -wasm-disable-explicit-locals -wasm-keep-registers -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling | FileCheck %s -check-prefixes=CHECK,NOPIC -DPTR=64
; RUN: llc < %s --mtriple=wasm32-unknown-emscripten -wasm-disable-explicit-locals -wasm-keep-registers -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling | FileCheck %s -check-prefixes=CHECK,NOPIC -DPTR=32
; RUN: llc < %s --mtriple=wasm64-unknown-emscripten -wasm-disable-explicit-locals -wasm-keep-registers -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling | FileCheck %s -check-prefixes=CHECK,NOPIC -DPTR=64
; RUN: llc < %s --mtriple=wasm32-unknown-emscripten -wasm-disable-explicit-locals -wasm-keep-registers -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling -relocation-model=pic | FileCheck %s -check-prefixes=CHECK,PIC -DPTR=32
; RUN: llc < %s --mtriple=wasm64-unknown-emscripten -wasm-disable-explicit-locals -wasm-keep-registers -wasm-enable-eh -exception-model=wasm -mattr=+exception-handling -relocation-model=pic | FileCheck %s -check-prefixes=CHECK,PIC -DPTR=64

@_ZTIi = external constant i8*
@_ZTIf = external constant i8*
@_ZTId = external constant i8*

; Single catch (...) does not need an exception table.
;
; try {
;   may_throw();
; } catch (...) {
; }
; CHECK-LABEL: test0:
; CHECK-NOT: GCC_except_table
define void @test0() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  invoke void @may_throw()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* null]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
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
; NOPIC:      i[[PTR]].const  $push[[CONTEXT:.*]]=, __wasm_lpad_context
; NOPIC-NEXT: i[[PTR]].const  $push[[EXCEPT_TABLE:.*]]=, GCC_except_table1
; NOPIC-NEXT: i[[PTR]].store  {{[48]}}($pop[[CONTEXT]]), $pop[[EXCEPT_TABLE]]

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
define void @test1() personality i8* bitcast (i32 (...)* @__gxx_wasm_personality_v0 to i8*) {
entry:
  invoke void @may_throw()
          to label %try.cont unwind label %catch.dispatch

catch.dispatch:                                   ; preds = %entry
  %0 = catchswitch within none [label %catch.start] unwind to caller

catch.start:                                      ; preds = %catch.dispatch
  %1 = catchpad within %0 [i8* bitcast (i8** @_ZTIi to i8*), i8* bitcast (i8** @_ZTIf to i8*), i8* bitcast (i8** @_ZTId to i8*), i8* null]
  %2 = call i8* @llvm.wasm.get.exception(token %1)
  %3 = call i32 @llvm.wasm.get.ehselector(token %1)
  %4 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches = icmp eq i32 %3, %4
  br i1 %matches, label %catch10, label %catch.fallthrough

catch10:                                          ; preds = %catch.start
  %5 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
  %6 = bitcast i8* %5 to i32*
  %7 = load i32, i32* %6, align 4
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

try.cont:                                         ; preds = %entry, %catch, %catch4, %catch7, %catch10
  invoke void @may_throw()
          to label %try.cont23 unwind label %catch.dispatch14

catch.dispatch14:                                 ; preds = %try.cont
  %8 = catchswitch within none [label %catch.start15] unwind to caller

catch.start15:                                    ; preds = %catch.dispatch14
  %9 = catchpad within %8 [i8* bitcast (i8** @_ZTId to i8*), i8* null]
  %10 = call i8* @llvm.wasm.get.exception(token %9)
  %11 = call i32 @llvm.wasm.get.ehselector(token %9)
  %12 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTId to i8*))
  %matches16 = icmp eq i32 %11, %12
  %13 = call i8* @__cxa_begin_catch(i8* %10) [ "funclet"(token %9) ]
  br i1 %matches16, label %catch20, label %catch17

catch20:                                          ; preds = %catch.start15
  %14 = bitcast i8* %13 to double*
  %15 = load double, double* %14, align 8
  call void @__cxa_end_catch() [ "funclet"(token %9) ]
  catchret from %9 to label %try.cont23

try.cont23:                                       ; preds = %try.cont, %catch17, %catch20
  invoke void @may_throw()
          to label %try.cont36 unwind label %catch.dispatch25

catch.dispatch25:                                 ; preds = %try.cont23
  %16 = catchswitch within none [label %catch.start26] unwind to caller

catch.start26:                                    ; preds = %catch.dispatch25
  %17 = catchpad within %16 [i8* bitcast (i8** @_ZTIi to i8*), i8* bitcast (i8** @_ZTIf to i8*)]
  %18 = call i8* @llvm.wasm.get.exception(token %17)
  %19 = call i32 @llvm.wasm.get.ehselector(token %17)
  %20 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIi to i8*))
  %matches27 = icmp eq i32 %19, %20
  br i1 %matches27, label %catch33, label %catch.fallthrough28

catch33:                                          ; preds = %catch.start26
  %21 = call i8* @__cxa_begin_catch(i8* %18) [ "funclet"(token %17) ]
  %22 = bitcast i8* %21 to i32*
  %23 = load i32, i32* %22, align 4
  call void @__cxa_end_catch() [ "funclet"(token %17) ]
  catchret from %17 to label %try.cont36

catch.fallthrough28:                              ; preds = %catch.start26
  %24 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIf to i8*))
  %matches29 = icmp eq i32 %19, %24
  br i1 %matches29, label %catch30, label %rethrow

catch30:                                          ; preds = %catch.fallthrough28
  %25 = call i8* @__cxa_begin_catch(i8* %18) [ "funclet"(token %17) ]
  %26 = bitcast i8* %25 to float*
  %27 = load float, float* %26, align 4
  call void @__cxa_end_catch() [ "funclet"(token %17) ]
  catchret from %17 to label %try.cont36

rethrow:                                          ; preds = %catch.fallthrough28
  call void @__cxa_rethrow() [ "funclet"(token %17) ]
  unreachable

try.cont36:                                       ; preds = %try.cont23, %catch30, %catch33
  ret void

catch17:                                          ; preds = %catch.start15
  call void @__cxa_end_catch() [ "funclet"(token %9) ]
  catchret from %9 to label %try.cont23

catch.fallthrough:                                ; preds = %catch.start
  %28 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTIf to i8*))
  %matches1 = icmp eq i32 %3, %28
  br i1 %matches1, label %catch7, label %catch.fallthrough2

catch7:                                           ; preds = %catch.fallthrough
  %29 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
  %30 = bitcast i8* %29 to float*
  %31 = load float, float* %30, align 4
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

catch.fallthrough2:                               ; preds = %catch.fallthrough
  %32 = call i32 @llvm.eh.typeid.for(i8* bitcast (i8** @_ZTId to i8*))
  %matches3 = icmp eq i32 %3, %32
  %33 = call i8* @__cxa_begin_catch(i8* %2) [ "funclet"(token %1) ]
  br i1 %matches3, label %catch4, label %catch

catch4:                                           ; preds = %catch.fallthrough2
  %34 = bitcast i8* %33 to double*
  %35 = load double, double* %34, align 8
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont

catch:                                            ; preds = %catch.fallthrough2
  call void @__cxa_end_catch() [ "funclet"(token %1) ]
  catchret from %1 to label %try.cont
}

declare void @may_throw()
; Function Attrs: nounwind
declare i32 @llvm.eh.typeid.for(i8*) #0
; Function Attrs: nounwind
declare i8* @llvm.wasm.get.exception(token) #0
; Function Attrs: nounwind
declare i32 @llvm.wasm.get.ehselector(token) #0
declare void @__cxa_rethrow()
declare i8* @__cxa_begin_catch(i8*)
declare void @__cxa_end_catch()
declare i32 @__gxx_wasm_personality_v0(...)

attributes #0 = { nounwind }
