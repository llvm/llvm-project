; The pass is enabled by default. At -O3 it does not reduce .rodata size, so
; every locally-defined global integer array (including multi-dimensional ones)
; is aligned to at least 8 bytes.
; RUN: llc -mtriple=hexagon -O3 \
; RUN:   -stop-after=hexagon-global-array-alignment < %s | FileCheck %s
;
; At -O1 and -O2 the pass reduces .rodata size, so byte and half-word arrays
; with alignment <= 2 keep their natural alignment, while word arrays are
; still aligned to 8 bytes.
; RUN: llc -mtriple=hexagon -O1 \
; RUN:   -stop-after=hexagon-global-array-alignment < %s \
; RUN:   | FileCheck %s --check-prefix=RODATA
; RUN: llc -mtriple=hexagon -O2 \
; RUN:   -stop-after=hexagon-global-array-alignment < %s \
; RUN:   | FileCheck %s --check-prefix=RODATA
;
; With -hexagon-disable-align-opt-byte-half the .rodata size reduction is off,
; so byte and half-word arrays are promoted to 8 bytes even at -O2, as at -O3.
; RUN: llc -mtriple=hexagon -O2 -hexagon-disable-align-opt-byte-half \
; RUN:   -stop-after=hexagon-global-array-alignment < %s | FileCheck %s
;
; The pass can be disabled with -hexagon-disable-global-array-align.
; RUN: llc -mtriple=hexagon -O3 -hexagon-disable-global-array-align \
; RUN:   -stop-after=hexagon-global-array-alignment < %s \
; RUN:   | FileCheck %s --check-prefix=DISABLED

; Locally-defined integer arrays are promoted to at least 8 bytes.
; CHECK: @int_array = dso_local global [4 x i32] zeroinitializer, align 8
; CHECK: @char_array = dso_local global [8 x i8] zeroinitializer, align 8
; CHECK: @short_array = dso_local global [4 x i16] zeroinitializer, align 8
; CHECK: @multidim = dso_local global [3 x [5 x i32]] zeroinitializer, align 8
; CHECK: @explicit = dso_local global [4 x i32] zeroinitializer, align 16
; CHECK: @scalar = dso_local global i32 0, align 4

; When reducing .rodata size, byte/half arrays at align <= 2 are left alone, but
; word arrays are still promoted to 8 bytes.
; RODATA: @int_array = dso_local global [4 x i32] zeroinitializer, align 8
; RODATA: @char_array = dso_local global [8 x i8] zeroinitializer, align 1
; RODATA: @short_array = dso_local global [4 x i16] zeroinitializer, align 2
; RODATA: @multidim = dso_local global [3 x [5 x i32]] zeroinitializer, align 8
; RODATA: @explicit = dso_local global [4 x i32] zeroinitializer, align 16
; RODATA: @scalar = dso_local global i32 0, align 4

; DISABLED: @int_array = dso_local global [4 x i32] zeroinitializer, align 4
; DISABLED: @char_array = dso_local global [8 x i8] zeroinitializer, align 1
; DISABLED: @short_array = dso_local global [4 x i16] zeroinitializer, align 2
; DISABLED: @multidim = dso_local global [3 x [5 x i32]] zeroinitializer, align 4
; DISABLED: @explicit = dso_local global [4 x i32] zeroinitializer, align 16
; DISABLED: @scalar = dso_local global i32 0, align 4

; The alignment of globals whose definition the linker may override or relocate
; is left untouched (see GlobalObject::canIncreaseAlignment). These must keep
; their original alignment for both the -O3 and -O2 runs.
; CHECK: @extern_decl = external dso_local global [4 x i32], align 4
; CHECK: @weak_arr = weak dso_local global [4 x i32] zeroinitializer, align 4
; CHECK: @interposable = global [4 x i32] zeroinitializer, align 4
; CHECK: @sectioned = dso_local global [4 x i32] zeroinitializer, section ".mysec", align 4
; RODATA: @extern_decl = external dso_local global [4 x i32], align 4
; RODATA: @weak_arr = weak dso_local global [4 x i32] zeroinitializer, align 4
; RODATA: @interposable = global [4 x i32] zeroinitializer, align 4
; RODATA: @sectioned = dso_local global [4 x i32] zeroinitializer, section ".mysec", align 4

@int_array = dso_local global [4 x i32] zeroinitializer, align 4
@char_array = dso_local global [8 x i8] zeroinitializer, align 1
@short_array = dso_local global [4 x i16] zeroinitializer, align 2
@multidim = dso_local global [3 x [5 x i32]] zeroinitializer, align 4
@explicit = dso_local global [4 x i32] zeroinitializer, align 16
@scalar = dso_local global i32 0, align 4

; A declaration: the definition (and its alignment) lives in another module.
@extern_decl = external dso_local global [4 x i32], align 4
; A weak definition: the linker may choose a different definition.
@weak_arr = weak dso_local global [4 x i32] zeroinitializer, align 4
; An interposable (non-dso_local) definition subject to COPY relocations.
@interposable = global [4 x i32] zeroinitializer, align 4
; A section-pinned global: raising the alignment could introduce padding.
@sectioned = dso_local global [4 x i32] zeroinitializer, section ".mysec", align 4
