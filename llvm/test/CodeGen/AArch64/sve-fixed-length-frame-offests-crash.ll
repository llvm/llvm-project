; RUN: llc < %s | FileCheck %s

target triple = "aarch64-unknown-linux-gnu"

; Ensure we don't crash by trying to fold fixed length frame indexes into
; loads/stores that don't support an appropriate addressing mode, hence creating
; too many extra vregs during frame lowering, when we don't have an emergency
; spill slot.

define dso_local void @func1(i64* %v1, i64* %v2, i64* %v3, i64* %v4, i64* %v5, i64* %v6, i64* %v7, i64* %v8,
                             i64* %v9, i64* %v10, i64* %v11, i64* %v12, i64* %v13, i64* %v14,  i64* %v15, i64* %v16,
                             i64* %v17, i64* %v18, i64* %v19, i64* %v20, i64* %v21, i64* %v22, i64* %v23, i64* %v24,
                             i64* %v25, i64* %v26, i64* %v27, i64* %v28, i64* %v29, i64* %v30, i64* %v31, i64* %v32,
                             i64* %v33, i64* %v34, i64* %v35, i64* %v36, i64* %v37, i64* %v38, i64* %v39, i64* %v40,
                             i64* %v41, i64* %v42, i64* %v43, i64* %v44, i64* %v45, i64* %v46, i64* %v47, i64* %v48,
                             i64 %v49) #0 {
; CHECK-LABEL: func1
  tail call void @func2(i64* %v1, i64* %v2, i64* %v3, i64* %v4, i64* %v5, i64* %v6, i64* %v7, i64* %v8,
                        i64* %v9, i64* %v10, i64* %v11, i64* %v12, i64* undef, i64* %v14, i64* %v15, i64* %v16,
                        i64* %v17, i64* %v18, i64* %v19, i64* %v20, i64* %v21, i64* %v22, i64* %v23, i64* %v24,
                        i64* %v25, i64* %v26, i64* %v27, i64* %v28, i64* %v29, i64* %v30, i64* undef, i64* undef,
                        i64* undef, i64* undef, i64* undef, i64* undef, i64* %v37, i64* %v38, i64* %v39, i64* %v40,
                        i64* %v41, i64* %v42, i64* %v43, i64* %v44, i64* %v45, i64* undef, i64* %v47, i64* %v48,
                        i64 undef)
  ret void
}

declare dso_local void @func2(i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*,
                              i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*,
                              i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*,
                              i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*,
                              i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*,
                              i64*, i64*, i64*, i64*, i64*, i64*, i64*, i64*,
                              i64)

attributes #0 = { "target-features"="+sve" vscale_range(2,2) }
