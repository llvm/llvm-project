; RUN: llvm-link %S/Inputs/opencl.md.a.ll %S/Inputs/opencl.md.c.ll -S | FileCheck %s

; OpenCL/SPIR version is different in input modules.
; Verify that different OpenCL/SPIR version metadata nodes are kept in the
; destination module.

; CHECK: !opencl.ocl.version = !{![[#MD0:]], ![[#MD1:]]}
; CHECK: !opencl.spir.version = !{![[#MD0]], ![[#MD1]]}
; CHECK: !opencl.used.extensions = !{![[#MD2:]], ![[#MD3:]]}
; CHECK: !opencl.used.optional.core.features = !{![[#MD4:]]}
; CHECK: !llvm.ident = !{![[#MD5:]]}
; CHECK: !llvm.module.flags = !{![[#MD6:]]}

; CHECK: ![[#MD0]] = !{i32 3, i32 0}
; CHECK: ![[#MD1]] = !{i32 2, i32 0}
; CHECK: ![[#MD2]] = !{!"cl_images", !"cl_khr_fp16"}
; CHECK: ![[#MD3]] = !{!"cl_images", !"cl_doubles"}
; CHECK: ![[#MD4]] = !{!"cl_images"}
; CHECK: ![[#MD5]] = !{!"LLVM.org clang version 20.1.0"}
; CHECK: ![[#MD6]] = !{i32 1, !"wchar_size", i32 4}
