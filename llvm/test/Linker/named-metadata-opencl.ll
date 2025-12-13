; RUN: llvm-link %S/Inputs/opencl.md.a.ll %S/Inputs/opencl.md.b.ll -S | FileCheck %s

; Verify that duplicate named metadata node isn't added to the result module.

; CHECK: !opencl.ocl.version = !{![[#MD0:]]}
; CHECK: !opencl.spir.version = !{![[#MD0]]}
; CHECK: !opencl.used.extensions = !{![[#MD1:]], ![[#MD2:]], ![[#MD3:]]}
; CHECK: !opencl.used.optional.core.features = !{![[#MD4:]]}
; CHECK: !llvm.ident = !{![[#MD5:]]}
; CHECK: !llvm.module.flags = !{![[#MD6:]]}

; CHECK: ![[#MD0]] = !{i32 3, i32 0}
; CHECK: ![[#MD1]] = !{!"cl_images", !"cl_khr_fp16"}
; CHECK: ![[#MD2]] = !{!"cl_images", !"cl_doubles"}
; CHECK: ![[#MD3]] = !{!"cl_khr_fp16", !"cl_doubles"}
; CHECK: ![[#MD4]] = !{!"cl_images"}
; CHECK: ![[#MD5]] = !{!"LLVM.org clang version 20.1.0"}
; CHECK: ![[#MD6]] = !{i32 1, !"wchar_size", i32 4}
