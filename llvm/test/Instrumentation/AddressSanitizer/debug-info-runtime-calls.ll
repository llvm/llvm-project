; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -asan-emit-debug-info -S | FileCheck %s --check-prefix=ENABLED
; RUN: opt < %s -passes=asan -asan-instrumentation-with-call-threshold=0 -S | FileCheck %s --check-prefix=DISABLED --implicit-check-not='declare !dbg'

declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1 immarg)

define void @test(ptr %src, ptr %dst) sanitize_address {
entry:
  %0 = load i32, ptr %src, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr %dst, ptr %src, i64 4, i1 false)
  ret void
}

; ENABLED: call void @__asan_load4(i64
; ENABLED: call ptr @__asan_memcpy(ptr %dst, ptr %src, i64 4)
; ENABLED: declare !dbg ![[LOADSP:[0-9]+]] void @__asan_load4(i64)
; ENABLED: declare !dbg ![[MEMCPYSP:[0-9]+]] ptr @__asan_memcpy(ptr, ptr, i64)
; ENABLED-DAG: ![[ASANFILE:[0-9]+]] = !DIFile(filename: "asan_interface.h", directory: "sanitizer")
; ENABLED-DAG: ![[LOADSP]] = !DISubprogram(name: "__asan_load4", linkageName: "__asan_load4", scope: ![[ASANFILE]], file: ![[ASANFILE]], type: ![[LOADTY:[0-9]+]]
; ENABLED-DAG: ![[LOADTY]] = !DISubroutineType(types: ![[LOADTYS:[0-9]+]])
; ENABLED-DAG: ![[LOADTYS]] = !{null, ![[I64:[0-9]+]]}
; ENABLED-DAG: ![[I64]] = !DIBasicType(name: "__int_64", size: 64, encoding: DW_ATE_signed, flags: DIFlagArtificial)
; ENABLED-DAG: ![[MEMCPYSP]] = !DISubprogram(name: "__asan_memcpy", linkageName: "__asan_memcpy", scope: ![[ASANFILE]], file: ![[ASANFILE]], type: ![[MEMCPYTY:[0-9]+]]
; ENABLED-DAG: ![[MEMCPYTY]] = !DISubroutineType(types: ![[MEMCPYTYS:[0-9]+]])
; ENABLED-DAG: ![[MEMCPYTYS]] = !{![[PTR:[0-9]+]], ![[PTR]], ![[PTR]], ![[I64]]}
; ENABLED-DAG: ![[PTR]] = !DIDerivedType(tag: DW_TAG_pointer_type{{.*}}size: 64

; DISABLED: call void @__asan_load4(i64
; DISABLED: call ptr @__asan_memcpy(ptr %dst, ptr %src, i64 4)
; DISABLED: declare void @__asan_load4(i64)
; DISABLED: declare ptr @__asan_memcpy(ptr, ptr, i64)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!3}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "asan-debug-info-runtime-calls.c", directory: "/tmp")
!3 = !{i32 2, !"Debug Info Version", i32 3}
