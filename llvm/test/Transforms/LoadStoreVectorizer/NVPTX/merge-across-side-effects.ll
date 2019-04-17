; RUN: opt -mtriple=nvptx64-nvidia-cuda -load-store-vectorizer -S -o - %s | FileCheck %s

; Check that the load/store vectorizer is willing to move loads/stores across
; intervening instructions only if it's safe.
;
;  - Loads can be moved across instructions that don't write or throw.
;  - Stores can only be moved across instructions which don't read, write, or
;    throw.

declare void @fn()
declare void @fn_nounwind() #0
declare void @fn_nounwind_writeonly() #1
declare void @fn_nounwind_readonly() #2
declare void @fn_writeonly() #3
declare void @fn_readonly() #4
declare void @fn_readnone() #5

; CHECK-LABEL: @load_fn
; CHECK: load
; CHECK: call void @fn()
; CHECK: load
define void @load_fn(i32* %p) #0 {
  %p.1 = getelementptr i32, i32* %p, i32 1

  %v0 = load i32, i32* %p, align 8
  call void @fn()
  %v1 = load i32, i32* %p.1, align 4
  ret void
}

; CHECK-LABEL: @load_fn_nounwind
; CHECK: load
; CHECK: call void @fn_nounwind()
; CHECK: load
define void @load_fn_nounwind(i32* %p) #0 {
  %p.1 = getelementptr i32, i32* %p, i32 1

  %v0 = load i32, i32* %p, align 8
  call void @fn_nounwind() #0
  %v1 = load i32, i32* %p.1, align 4
  ret void
}

; CHECK-LABEL: @load_fn_nounwind_writeonly
; CHECK: load
; CHECK: call void @fn_nounwind_writeonly()
; CHECK: load
define void @load_fn_nounwind_writeonly(i32* %p) #0 {
  %p.1 = getelementptr i32, i32* %p, i32 1

  %v0 = load i32, i32* %p, align 8
  call void @fn_nounwind_writeonly() #1
  %v1 = load i32, i32* %p.1, align 4
  ret void
}

; CHECK-LABEL: @load_fn_nounwind_readonly
; CHECK-DAG: load <2 x i32>
; CHECK-DAG: call void @fn_nounwind_readonly()
define void @load_fn_nounwind_readonly(i32* %p) #0 {
  %p.1 = getelementptr i32, i32* %p, i32 1

  %v0 = load i32, i32* %p, align 8
  call void @fn_nounwind_readonly() #2
  %v1 = load i32, i32* %p.1, align 4
  ret void
}

; CHECK-LABEL: @load_fn_readonly
; CHECK: load
; CHECK: call void @fn_readonly
; CHECK: load
define void @load_fn_readonly(i32* %p) #0 {
  %p.1 = getelementptr i32, i32* %p, i32 1

  %v0 = load i32, i32* %p, align 8
  call void @fn_readonly() #4
  %v1 = load i32, i32* %p.1, align 4
  ret void
}

; CHECK-LABEL: @load_fn_writeonly
; CHECK: load
; CHECK: call void @fn_writeonly()
; CHECK: load
define void @load_fn_writeonly(i32* %p) #0 {
  %p.1 = getelementptr i32, i32* %p, i32 1

  %v0 = load i32, i32* %p, align 8
  call void @fn_writeonly() #3
  %v1 = load i32, i32* %p.1, align 4
  ret void
}

; CHECK-LABEL: @load_fn_readnone
; CHECK-DAG: load <2 x i32>
; CHECK-DAG: call void @fn_readnone()
define void @load_fn_readnone(i32* %p) #0 {
  %p.1 = getelementptr i32, i32* %p, i32 1

  %v0 = load i32, i32* %p, align 8
  call void @fn_readnone() #5
  %v1 = load i32, i32* %p.1, align 4
  ret void
}

; ------------------------------------------------
; Same tests, but now for stores instead of loads.
; ------------------------------------------------

; CHECK-LABEL: @store_fn
; CHECK: store
; CHECK: call void @fn()
; CHECK: store
define void @store_fn(i32* %p) #0 {
  %p.1 = getelementptr i32, i32* %p, i32 1

  store i32 0, i32* %p
  call void @fn()
  store i32 0, i32* %p.1
  ret void
}

; CHECK-LABEL: @store_fn_nounwind
; CHECK: store
; CHECK: call void @fn_nounwind()
; CHECK: store
define void @store_fn_nounwind(i32* %p) #0 {
  %p.1 = getelementptr i32, i32* %p, i32 1

  store i32 0, i32* %p
  call void @fn_nounwind() #0
  store i32 0, i32* %p.1
  ret void
}

; CHECK-LABEL: @store_fn_nounwind_writeonly
; CHECK: store
; CHECK: call void @fn_nounwind_writeonly()
; CHECK: store
define void @store_fn_nounwind_writeonly(i32* %p) #0 {
  %p.1 = getelementptr i32, i32* %p, i32 1

  store i32 0, i32* %p
  call void @fn_nounwind_writeonly() #1
  store i32 0, i32* %p.1
  ret void
}

; CHECK-LABEL: @store_fn_nounwind_readonly
; CHECK: store
; CHECK: call void @fn_nounwind_readonly()
; CHECK: store
define void @store_fn_nounwind_readonly(i32* %p) #0 {
  %p.1 = getelementptr i32, i32* %p, i32 1

  store i32 0, i32* %p
  call void @fn_nounwind_readonly() #2
  store i32 0, i32* %p.1
  ret void
}

; CHECK-LABEL: @store_fn_readonly
; CHECK: store
; CHECK: call void @fn_readonly
; CHECK: store
define void @store_fn_readonly(i32* %p) #0 {
  %p.1 = getelementptr i32, i32* %p, i32 1

  store i32 0, i32* %p
  call void @fn_readonly() #4
  store i32 0, i32* %p.1
  ret void
}

; CHECK-LABEL: @store_fn_writeonly
; CHECK: store
; CHECK: call void @fn_writeonly()
; CHECK: store
define void @store_fn_writeonly(i32* %p) #0 {
  %p.1 = getelementptr i32, i32* %p, i32 1

  store i32 0, i32* %p
  call void @fn_writeonly() #3
  store i32 0, i32* %p.1
  ret void
}

; This is the only store idiom we can vectorize.
; CHECK-LABEL: @store_fn_readnone
; CHECK-DAG: store <2 x i32>
; CHECK-DAG: call void @fn_readnone()
define void @store_fn_readnone(i32* %p) #0 {
  %p.1 = getelementptr i32, i32* %p, i32 1

  store i32 0, i32* %p, align 8
  call void @fn_readnone() #5
  store i32 0, i32* %p.1, align 8
  ret void
}


attributes #0 = { nounwind }
attributes #1 = { nounwind writeonly }
attributes #2 = { nounwind readonly }
attributes #3 = { writeonly }
attributes #4 = { readonly }
; readnone implies nounwind, so no need to test separately
attributes #5 = { nounwind readnone }
