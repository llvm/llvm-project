; RUN: llc -mtriple=hexagon < %s | FileCheck %s

; CHECK-LABEL: l48:
; CHECK: v{{.+}} = vmem({{.+}})
; CHECK-DAG: v{{.+}} = vmem({{.+}})
; CHECK-DAG: vmux
; CHECK: vmux
define <48 x i32> @l48(ptr align 128 %p, <48 x i32> %v, <48 x i32> %a) #0 {
  %q = icmp sgt <48 x i32> %a, zeroinitializer
  %r = call <48 x i32> @llvm.masked.load.vv80i32.p0(ptr %p, <48 x i1> %q, <48 x i32> %v)
  ret <48 x i32> %r
}

; CHECK-LABEL: l64:
; CHECK: v{{.+}} = vmem({{.+}})
; CHECK-DAG: v{{.+}} = vmem({{.+}})
; CHECK-DAG: vmux
; CHECK: vmux
define <64 x i32> @l64(ptr align 128 %p, <64 x i32> %v, <64 x i32> %a) #0 {
  %q = icmp sgt <64 x i32> %a, zeroinitializer
  %r = call <64 x i32> @llvm.masked.load.vv80i32.p0(ptr %p, <64 x i1> %q, <64 x i32> %v)
  ret <64 x i32> %r
}

; CHECK-LABEL: l80:
; CHECK: v{{.+}} = vmem({{.+}})
; CHECK-DAG: v{{.+}} = vmem({{.+}})
; CHECK-DAG: v{{.+}} = vmem({{.+}})
; CHECK-DAG: vmux
; CHECK-DAG: vmux
; CHECK: vmux
define <80 x i32> @l80(ptr align 128 %p, <80 x i32> %v, <80 x i32> %a) #0 {
  %q = icmp sgt <80 x i32> %a, zeroinitializer
  %r = call <80 x i32> @llvm.masked.load.vv80i32.p0(ptr %p, <80 x i1> %q, <80 x i32> %v)
  ret <80 x i32> %r
}

; CHECK-LABEL: s48:
; CHECK: if (q{{.}}) vmem{{.*}} = v
; CHECK: if (q{{.}}) vmem{{.*}} = v
; CHECK-NOT: vmem
define void @s48(ptr %p, <48 x i32> %v, <48 x i32> %a) #0 {
  %q = icmp sgt <48 x i32> %a, zeroinitializer
  call void @llvm.masked.store.vv80i32.p0(<48 x i32> %v, ptr %p, i32 128, <48 x i1> %q)
  ret void
}

; CHECK-LABEL: s64:
; CHECK: if (q{{.}}) vmem{{.*}} = v
; CHECK: if (q{{.}}) vmem{{.*}} = v
; CHECK-NOT: vmem
define void @s64(ptr %p, <64 x i32> %v, <64 x i32> %a) #0 {
  %q = icmp sgt <64 x i32> %a, zeroinitializer
  call void @llvm.masked.store.vv80i32.p0(<64 x i32> %v, ptr %p, i32 128, <64 x i1> %q)
  ret void
}

; CHECK-LABEL: s80:
; CHECK: if (q{{.}}) vmem{{.*}} = v
; CHECK: if (q{{.}}) vmem{{.*}} = v
; CHECK: if (q{{.}}) vmem{{.*}} = v
; CHECK-NOT: vmem
define void @s80(ptr %p, <80 x i32> %v, <80 x i32> %a) #0 {
  %q = icmp sgt <80 x i32> %a, zeroinitializer
  call void @llvm.masked.store.vv80i32.p0(<80 x i32> %v, ptr %p, i32 128, <80 x i1> %q)
  ret void
}

attributes #0 = { nounwind "target-features"="+hvxv73,+hvx-length128b" }
