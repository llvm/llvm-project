; RUN: opt -passes=inferattrs -S %s | FileCheck %s

; The library functions return objects of exactly the requested size, also if
; their declarations come with an `allocsize` that only gives a lower bound.
; Other functions keep the attribute that they have.

; CHECK: declare noalias noundef ptr @malloc(i64 noundef) #[[MALLOC:[0-9]+]]
declare ptr @malloc(i64) allocsize(0)
; CHECK: declare noalias noundef ptr @calloc(i64 noundef, i64 noundef) #[[CALLOC:[0-9]+]]
declare ptr @calloc(i64, i64) allocsize(0, 1)
; CHECK: declare ptr @my_alloc(i64) #[[MYALLOC:[0-9]+]]
declare ptr @my_alloc(i64) allocsize(0)

; CHECK-DAG: attributes #[[MALLOC]] = { {{.*}}allocsize(0,exact){{.*}} }
; CHECK-DAG: attributes #[[CALLOC]] = { {{.*}}allocsize(0,1,exact){{.*}} }
; CHECK-DAG: attributes #[[MYALLOC]] = { allocsize(0) }
