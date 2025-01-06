; RUN: llc -mtriple aarch64-linux-gnu    -mattr=+pauth -filetype=asm -o - %s | FileCheck --check-prefix=ELF %s
; RUN: llc -mtriple aarch64-apple-darwin -mattr=+pauth -filetype=asm -o - %s | FileCheck --check-prefix=MACHO %s

; ELF-LABEL:   _ZTI10Disc:
; ELF-NEXT:      .xword  (_ZTVN10__cxxabiv117__class_type_infoE+16)@AUTH(da,45546,addr)
; ELF-LABEL:   _ZTI10NoDisc:
; ELF-NEXT:      .xword  (_ZTVN10__cxxabiv117__class_type_infoE+16)@AUTH(da,45546)

; MACHO-LABEL: __ZTI10Disc:
; MACHO-NEXT:    .quad   (__ZTVN10__cxxabiv117__class_type_infoE+16)@AUTH(da,45546,addr)
; MACHO-LABEL: __ZTI10NoDisc:
; MACHO-NEXT:    .quad   (__ZTVN10__cxxabiv117__class_type_infoE+16)@AUTH(da,45546)


@_ZTI10Disc   = constant { ptr, ptr } { ptr ptrauth (ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), i32 2, i64 45546, ptr @_ZTI10Disc), ptr @_ZTS10Disc }, align 8
@_ZTS10Disc   = constant [4 x i8] c"Disc", align 1

@_ZTI10NoDisc = constant { ptr, ptr } { ptr ptrauth (ptr getelementptr inbounds (ptr, ptr @_ZTVN10__cxxabiv117__class_type_infoE, i64 2), i32 2, i64 45546), ptr @_ZTS10NoDisc }, align 8
@_ZTS10NoDisc = constant [6 x i8] c"NoDisc", align 1

@_ZTVN10__cxxabiv117__class_type_infoE = external global [0 x ptr]
