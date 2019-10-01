; RUN: opt -disable-output -licm -print-memoryssa -enable-mssa-loop-dependency=true < %s 2>&1 | FileCheck %s
@v_274 = external dso_local global i64, align 1
@v_295 = external dso_local global i16, align 1
@v_335 = external dso_local global i32, align 1

; CHECK-LABEL: @main()
; CHECK-NOT: 5 = MemoryPhi(
; CHECK-NOT: 6 = MemoryPhi(
; CHECK: 4 = MemoryPhi(
; CHECK-NOT: 7 = MemoryPhi(
define dso_local void @main() {
entry:
  store i32 undef, i32* @v_335, align 1
  br i1 undef, label %gate, label %exit

nopredentry1:                                     ; No predecessors!
  br label %preinfiniteloop

nopredentry2:                                     ; No predecessors!
  br label %gate

gate:                                             ; preds = %nopredentry2, %entry
  br i1 undef, label %preinfiniteloop, label %exit

preinfiniteloop:                                  ; preds = %gate, %nopredentry1
  br label %infiniteloop

infiniteloop:                                     ; preds = %infiniteloop, %preinfiniteloop
  store i16 undef, i16* @v_295, align 1
  br label %infiniteloop

exit:                                             ; preds = %gate, %entry
  store i64 undef, i64* @v_274, align 1
  ret void
}
