; RUN: llc -march=mips -mattr=+msa,+fp64 -relocation-model=pic \
; RUN:   -verify-machineinstrs < %s | \
; RUN:   FileCheck -check-prefixes=ALL,O32,MIPS32,ALL-BE,O32-BE %s
; RUN: llc -march=mipsel -mattr=+msa,+fp64 -relocation-model=pic \
; RUN:   -verify-machineinstrs < %s | \
; RUN:   FileCheck -check-prefixes=ALL,O32,MIPS32,ALL-LE,O32-LE %s
; RUN: llc -march=mips64 -target-abi n32 -mattr=+msa,+fp64 \
; RUN:   -relocation-model=pic -verify-machineinstrs < %s | \
; RUN:   FileCheck -check-prefixes=ALL,N32,MIPS64,ALL-BE %s
; RUN: llc -march=mips64el -target-abi n32 -mattr=+msa,+fp64 \
; RUN:   -relocation-model=pic -verify-machineinstrs < %s | \
; RUN:   FileCheck -check-prefixes=ALL,N32,MIPS64,ALL-LE %s
; RUN: llc -march=mips64 -mattr=+msa,+fp64 -relocation-model=pic \
; RUN:   -verify-machineinstrs < %s | \
; RUN:   FileCheck -check-prefixes=ALL,N64,MIPS64,ALL-BE %s
; RUN: llc -march=mips64el -mattr=+msa,+fp64 -relocation-model=pic \
; RUN:   -verify-machineinstrs < %s | \
; RUN:   FileCheck -check-prefixes=ALL,N64,MIPS64,ALL-LE %s

@v4i8 = global <4 x i8> <i8 0, i8 0, i8 0, i8 0>
@v16i8 = global <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>
@v8i16 = global <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>
@v4i32 = global <4 x i32> <i32 0, i32 0, i32 0, i32 0>
@v2i64 = global <2 x i64> <i64 0, i64 0>
@i32 = global i32 0
@i64 = global i64 0

define void @const_v16i8() nounwind {
  ; ALL-LABEL: const_v16i8:

  store volatile <16 x i8> <i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0, i8 0>, <16 x i8>*@v16i8
  ; ALL: ldi.b [[R1:\$w[0-9]+]], 0

  store volatile <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>, <16 x i8>*@v16i8
  ; ALL: ldi.b [[R1:\$w[0-9]+]], 1

  store volatile <16 x i8> <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 31>, <16 x i8>*@v16i8
  ; O32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; N32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst(.L
  ; N64: daddiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst(.L
  ; ALL: ld.b  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <16 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 9, i8 0, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6>, <16 x i8>*@v16i8
  ; O32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; N32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst(.L
  ; N64: daddiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst(.L
  ; ALL: ld.b  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <16 x i8> <i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0, i8 1, i8 0>, <16 x i8>*@v16i8
  ; ALL-BE: ldi.h [[R1:\$w[0-9]+]], 256
  ; ALL-LE: ldi.h [[R1:\$w[0-9]+]], 1

  store volatile <16 x i8> <i8 1, i8 2, i8 3, i8 4, i8 1, i8 2, i8 3, i8 4, i8 1, i8 2, i8 3, i8 4, i8 1, i8 2, i8 3, i8 4>, <16 x i8>*@v16i8
  ; ALL-BE-DAG: lui [[R2:\$[0-9]+]], 258
  ; ALL-LE-DAG: lui [[R2:\$[0-9]+]], 1027
  ; ALL-BE-DAG: ori [[R2]], [[R2]], 772
  ; ALL-LE-DAG: ori [[R2]], [[R2]], 513
  ; ALL-DAG: fill.w [[R1:\$w[0-9]+]], [[R2]]

  store volatile <16 x i8> <i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8, i8 1, i8 2, i8 3, i8 4, i8 5, i8 6, i8 7, i8 8>, <16 x i8>*@v16i8
  ; ALL-BE-DAG: lui [[R3:\$[0-9]+]], 1286
  ; ALL-LE-DAG: lui [[R3:\$[0-9]+]], 2055
  ; ALL-BE-DAG: ori [[R4:\$[0-9]+]], [[R3]], 1800
  ; ALL-LE-DAG: ori [[R4:\$[0-9]+]], [[R3]], 1541
  ; O32-BE: fill.w  [[R1:\$w[0-9]+]], [[R4]]

  ; O32: insert.w [[R1]][1], [[R2]]
  ; O32: splati.d $w{{.*}}, [[R1]][0]

  ; MIPS64-BE: dinsu [[R4]], [[R2]], 32, 32
  ; MIPS64-LE: dinsu [[R2]], [[R4]], 32, 32
  ; MIPS64-BE: fill.d $w{{.*}}, [[R4]]
  ; MIPS64-LE: fill.d $w{{.*}}, [[R2]]

  ret void
}

define void @const_v8i16() nounwind {
  ; ALL-LABEL: const_v8i16:

  store volatile <8 x i16> <i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0, i16 0>, <8 x i16>*@v8i16
  ; ALL: ldi.b [[R1:\$w[0-9]+]], 0

  store volatile <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>, <8 x i16>*@v8i16
  ; ALL: ldi.h [[R1:\$w[0-9]+]], 1

  store volatile <8 x i16> <i16 1, i16 1, i16 1, i16 2, i16 1, i16 1, i16 1, i16 31>, <8 x i16>*@v8i16
  ; O32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; N32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst(.L
  ; N64: daddiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst(.L
  ; ALL: ld.h  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <8 x i16> <i16 1028, i16 1028, i16 1028, i16 1028, i16 1028, i16 1028, i16 1028, i16 1028>, <8 x i16>*@v8i16
  ; ALL: ldi.b [[R1:\$w[0-9]+]], 4

  store volatile <8 x i16> <i16 1, i16 2, i16 1, i16 2, i16 1, i16 2, i16 1, i16 2>, <8 x i16>*@v8i16
  ; ALL-BE-DAG: lui [[R2:\$[0-9]+]], 1
  ; ALL-LE-DAG: lui [[R2:\$[0-9]+]], 2
  ; ALL-BE-DAG: ori [[R2]], [[R2]], 2
  ; ALL-LE-DAG: ori [[R2]], [[R2]], 1
  ; ALL-DAG: fill.w [[R1:\$w[0-9]+]], [[R2]]

  store volatile <8 x i16> <i16 1, i16 2, i16 3, i16 4, i16 1, i16 2, i16 3, i16 4>, <8 x i16>*@v8i16
  ; ALL-BE-DAG: lui [[R3:\$[0-9]+]], 3
  ; ALL-LE-DAG: lui [[R3:\$[0-9]+]], 4
  ; ALL-BE-DAG: ori [[R4:\$[0-9]+]], [[R3]], 4
  ; ALL-LE-DAG: ori [[R4:\$[0-9]+]], [[R3]], 3

  ; O32-BE: fill.w [[R1:\$w[0-9]+]], [[R4]]
  ; O32: insert.w [[R1]][1], [[R2]]
  ; O32: splati.d $w{{.*}}, [[R1]][0]

  ; MIPS64-BE: dinsu [[R4]], [[R2]], 32, 32
  ; MIPS64-LE: dinsu [[R2]], [[R4]], 32, 32
  ; MIPS64-BE: fill.d $w{{.*}}, [[R4]]
  ; MIPS64-LE: fill.d $w{{.*}}, [[R2]]

  ret void
}

define void @const_v4i32() nounwind {
  ; ALL-LABEL: const_v4i32:

  store volatile <4 x i32> <i32 0, i32 0, i32 0, i32 0>, <4 x i32>*@v4i32
  ; ALL: ldi.b [[R1:\$w[0-9]+]], 0

  store volatile <4 x i32> <i32 1, i32 1, i32 1, i32 1>, <4 x i32>*@v4i32
  ; ALL: ldi.w [[R1:\$w[0-9]+]], 1

  store volatile <4 x i32> <i32 1, i32 1, i32 1, i32 31>, <4 x i32>*@v4i32
  ; O32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; N32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst(.L
  ; N64: daddiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst(.L
  ; ALL: ld.w  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <4 x i32> <i32 16843009, i32 16843009, i32 16843009, i32 16843009>, <4 x i32>*@v4i32
  ; ALL: ldi.b [[R1:\$w[0-9]+]], 1

  store volatile <4 x i32> <i32 65537, i32 65537, i32 65537, i32 65537>, <4 x i32>*@v4i32
  ; ALL: ldi.h [[R1:\$w[0-9]+]], 1

  store volatile <4 x i32> <i32 1, i32 2, i32 1, i32 2>, <4 x i32>*@v4i32
  ; -BE-DAG: ori [[R2:\$[0-9]+]], $zero, 1
  ; O32-BE-DAG: ori [[R3:\$[0-9]+]], $zero, 1
  ; O32-BE-DAG: ori [[R4:\$[0-9]+]], $zero, 2
  ; O32-LE-DAG: ori [[R3:\$[0-9]+]], $zero, 2
  ; O32-LE-DAG: ori [[R4:\$[0-9]+]], $zero, 1
  ; O32: fill.w [[W0:\$w[0-9]+]], [[R4]]
  ; O32: insert.w [[W0]][1], [[R3]]
  ; O32: splati.d [[W1:\$w[0-9]+]], [[W0]]

  ; MIPS64-DAG: ori [[R5:\$[0-9]+]], $zero, 2
  ; MIPS64-DAG: ori [[R6:\$[0-9]+]], $zero, 1

  ; MIPS64-BE: dinsu [[R5]], [[R6]], 32, 32
  ; MIPS64-LE: dinsu [[R6]], [[R5]], 32, 32
  ; MIPS64-BE: fill.d $w{{.*}}, [[R4]]
  ; MIPS64-LE: fill.d $w{{.*}}, [[R2]]


  store volatile <4 x i32> <i32 3, i32 4, i32 5, i32 6>, <4 x i32>*@v4i32
  ; O32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; N32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst(.L
  ; N64: daddiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst(.L
  ; ALL: ld.w  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  ret void
}

define void @const_v2i64() nounwind {
  ; ALL-LABEL: const_v2i64:

  store volatile <2 x i64> <i64 0, i64 0>, <2 x i64>*@v2i64
  ; ALL: ldi.b [[R1:\$w[0-9]+]], 0

  store volatile <2 x i64> <i64 72340172838076673, i64 72340172838076673>, <2 x i64>*@v2i64
  ; ALL: ldi.b [[R1:\$w[0-9]+]], 1

  store volatile <2 x i64> <i64 281479271743489, i64 281479271743489>, <2 x i64>*@v2i64
  ; ALL: ldi.h [[R1:\$w[0-9]+]], 1

  store volatile <2 x i64> <i64 4294967297, i64 4294967297>, <2 x i64>*@v2i64
  ; ALL: ldi.w [[R1:\$w[0-9]+]], 1

  store volatile <2 x i64> <i64 1, i64 1>, <2 x i64>*@v2i64
  ; ALL: ldi.d [[R1:\$w[0-9]+]], 1

  store volatile <2 x i64> <i64 1, i64 31>, <2 x i64>*@v2i64
  ; O32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; N32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst(.L
  ; N64: daddiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst(.L
  ; MIPS32: ld.w [[R1:\$w[0-9]+]], 0([[G_PTR]])
  ; MIPS64: ld.d [[R1:\$w[0-9]+]], 0([[G_PTR]])

  store volatile <2 x i64> <i64 3, i64 4>, <2 x i64>*@v2i64
  ; O32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %lo($
  ; N32: addiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst(.L
  ; N64: daddiu [[G_PTR:\$[0-9]+]], {{.*}}, %got_ofst(.L
  ; MIPS32: ld.w  [[R1:\$w[0-9]+]], 0([[G_PTR]])
  ; MIPS64: ld.d  [[R1:\$w[0-9]+]], 0([[G_PTR]])

  ret void
}

define void @nonconst_v16i8(i8 signext %a, i8 signext %b, i8 signext %c, i8 signext %d, i8 signext %e, i8 signext %f, i8 signext %g, i8 signext %h) nounwind {
  ; ALL-LABEL: nonconst_v16i8:

  %1 = insertelement <16 x i8> undef, i8 %a, i32 0
  %2 = insertelement <16 x i8> %1, i8 %b, i32 1
  %3 = insertelement <16 x i8> %2, i8 %c, i32 2
  %4 = insertelement <16 x i8> %3, i8 %d, i32 3
  %5 = insertelement <16 x i8> %4, i8 %e, i32 4
  %6 = insertelement <16 x i8> %5, i8 %f, i32 5
  %7 = insertelement <16 x i8> %6, i8 %g, i32 6
  %8 = insertelement <16 x i8> %7, i8 %h, i32 7
  %9 = insertelement <16 x i8> %8, i8 %h, i32 8
  %10 = insertelement <16 x i8> %9, i8 %h, i32 9
  %11 = insertelement <16 x i8> %10, i8 %h, i32 10
  %12 = insertelement <16 x i8> %11, i8 %h, i32 11
  %13 = insertelement <16 x i8> %12, i8 %h, i32 12
  %14 = insertelement <16 x i8> %13, i8 %h, i32 13
  %15 = insertelement <16 x i8> %14, i8 %h, i32 14
  %16 = insertelement <16 x i8> %15, i8 %h, i32 15
  ; ALL-DAG: insert.b [[R1:\$w[0-9]+]][0], $4
  ; ALL-DAG: insert.b [[R1]][1], $5
  ; ALL-DAG: insert.b [[R1]][2], $6
  ; ALL-DAG: insert.b [[R1]][3], $7
  ; MIPS32-DAG: lw [[R2:\$[0-9]+]], 16($sp)
  ; MIPS32-DAG: insert.b [[R1]][4], [[R2]]
  ; MIPS64-DAG: insert.b [[R1]][4], $8
  ; MIPS32-DAG: lw [[R3:\$[0-9]+]], 20($sp)
  ; MIPS32-DAG: insert.b [[R1]][5], [[R3]]
  ; MIPS64-DAG: insert.b [[R1]][5], $9
  ; MIPS32-DAG: lw [[R4:\$[0-9]+]], 24($sp)
  ; MIPS32-DAG: insert.b [[R1]][6], [[R4]]
  ; MIPS64-DAG: insert.b [[R1]][6], $10
  ; MIPS32-DAG: lw [[R5:\$[0-9]+]], 28($sp)
  ; MIPS32-DAG: insert.b [[R1]][7], [[R5]]
  ; MIPS64-DAG: insert.b [[R1]][7], [[R5:\$11]]
  ; ALL-DAG: insert.b [[R1]][8], [[R5]]
  ; ALL-DAG: insert.b [[R1]][9], [[R5]]
  ; ALL-DAG: insert.b [[R1]][10], [[R5]]
  ; ALL-DAG: insert.b [[R1]][11], [[R5]]
  ; ALL-DAG: insert.b [[R1]][12], [[R5]]
  ; ALL-DAG: insert.b [[R1]][13], [[R5]]
  ; ALL-DAG: insert.b [[R1]][14], [[R5]]
  ; ALL-DAG: insert.b [[R1]][15], [[R5]]

  store volatile <16 x i8> %16, <16 x i8>*@v16i8

  ret void
}

define void @nonconst_v8i16(i16 signext %a, i16 signext %b, i16 signext %c, i16 signext %d, i16 signext %e, i16 signext %f, i16 signext %g, i16 signext %h) nounwind {
  ; ALL-LABEL: nonconst_v8i16:

  %1 = insertelement <8 x i16> undef, i16 %a, i32 0
  %2 = insertelement <8 x i16> %1, i16 %b, i32 1
  %3 = insertelement <8 x i16> %2, i16 %c, i32 2
  %4 = insertelement <8 x i16> %3, i16 %d, i32 3
  %5 = insertelement <8 x i16> %4, i16 %e, i32 4
  %6 = insertelement <8 x i16> %5, i16 %f, i32 5
  %7 = insertelement <8 x i16> %6, i16 %g, i32 6
  %8 = insertelement <8 x i16> %7, i16 %h, i32 7
  ; ALL-DAG: insert.h [[R1:\$w[0-9]+]][0], $4
  ; ALL-DAG: insert.h [[R1]][1], $5
  ; ALL-DAG: insert.h [[R1]][2], $6
  ; ALL-DAG: insert.h [[R1]][3], $7
  ; MIPS32-DAG: lw [[R2:\$[0-9]+]], 16($sp)
  ; MIPS32-DAG: insert.h [[R1]][4], [[R2]]
  ; MIPS64-DAG: insert.h [[R1]][4], $8
  ; MIPS32-DAG: lw [[R2:\$[0-9]+]], 20($sp)
  ; MIPS32-DAG: insert.h [[R1]][5], [[R2]]
  ; MIPS64-DAG: insert.h [[R1]][5], $9
  ; MIPS32-DAG: lw [[R2:\$[0-9]+]], 24($sp)
  ; MIPS32-DAG: insert.h [[R1]][6], [[R2]]
  ; MIPS64-DAG: insert.h [[R1]][6], $10
  ; MIPS32-DAG: lw [[R2:\$[0-9]+]], 28($sp)
  ; MIPS32-DAG: insert.h [[R1]][7], [[R2]]
  ; MIPS64-DAG: insert.h [[R1]][7], $11

  store volatile <8 x i16> %8, <8 x i16>*@v8i16

  ret void
}

define void @nonconst_v4i32(i32 signext %a, i32 signext %b, i32 signext %c, i32 signext %d) nounwind {
  ; ALL-LABEL: nonconst_v4i32:

  %1 = insertelement <4 x i32> undef, i32 %a, i32 0
  %2 = insertelement <4 x i32> %1, i32 %b, i32 1
  %3 = insertelement <4 x i32> %2, i32 %c, i32 2
  %4 = insertelement <4 x i32> %3, i32 %d, i32 3
  ; ALL: insert.w [[R1:\$w[0-9]+]][0], $4
  ; ALL: insert.w [[R1]][1], $5
  ; ALL: insert.w [[R1]][2], $6
  ; ALL: insert.w [[R1]][3], $7

  store volatile <4 x i32> %4, <4 x i32>*@v4i32

  ret void
}

define void @nonconst_v2i64(i64 signext %a, i64 signext %b) nounwind {
  ; ALL-LABEL: nonconst_v2i64:

  %1 = insertelement <2 x i64> undef, i64 %a, i32 0
  %2 = insertelement <2 x i64> %1, i64 %b, i32 1
  ; MIPS32: insert.w [[R1:\$w[0-9]+]][0], $4
  ; MIPS32: insert.w [[R1]][1], $5
  ; MIPS32: insert.w [[R1]][2], $6
  ; MIPS32: insert.w [[R1]][3], $7
  ; MIPS64: insert.d [[R1:\$w[0-9]+]][0], $4
  ; MIPS64: insert.d [[R1]][1], $5

  store volatile <2 x i64> %2, <2 x i64>*@v2i64

  ret void
}

define i32 @extract_sext_v16i8() nounwind {
  ; ALL-LABEL: extract_sext_v16i8:

  %1 = load <16 x i8>, <16 x i8>* @v16i8
  ; ALL-DAG: ld.b [[R1:\$w[0-9]+]],

  %2 = add <16 x i8> %1, %1
  ; ALL-DAG: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <16 x i8> %2, i32 1
  %4 = sext i8 %3 to i32
  ; ALL-DAG: copy_s.b [[R3:\$[0-9]+]], [[R1]][1]
  ; ALL-NOT: sll
  ; ALL-NOT: sra

  ret i32 %4
}

define i32 @extract_sext_v8i16() nounwind {
  ; ALL-LABEL: extract_sext_v8i16:

  %1 = load <8 x i16>, <8 x i16>* @v8i16
  ; ALL-DAG: ld.h [[R1:\$w[0-9]+]],

  %2 = add <8 x i16> %1, %1
  ; ALL-DAG: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <8 x i16> %2, i32 1
  %4 = sext i16 %3 to i32
  ; ALL-DAG: copy_s.h [[R3:\$[0-9]+]], [[R1]][1]
  ; ALL-NOT: sll
  ; ALL-NOT: sra

  ret i32 %4
}

define i32 @extract_sext_v4i32() nounwind {
  ; ALL-LABEL: extract_sext_v4i32:

  %1 = load <4 x i32>, <4 x i32>* @v4i32
  ; ALL-DAG: ld.w [[R1:\$w[0-9]+]],

  %2 = add <4 x i32> %1, %1
  ; ALL-DAG: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <4 x i32> %2, i32 1
  ; ALL-DAG: copy_s.w [[R3:\$[0-9]+]], [[R1]][1]

  ret i32 %3
}

define i64 @extract_sext_v2i64() nounwind {
  ; ALL-LABEL: extract_sext_v2i64:

  %1 = load <2 x i64>, <2 x i64>* @v2i64
  ; ALL-DAG: ld.d [[R1:\$w[0-9]+]],

  %2 = add <2 x i64> %1, %1
  ; ALL-DAG: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <2 x i64> %2, i32 1
  ; MIPS32-DAG: copy_s.w [[R3:\$[0-9]+]], [[R1]][2]
  ; MIPS32-DAG: copy_s.w [[R4:\$[0-9]+]], [[R1]][3]
  ; MIPS64-DAG: copy_s.d [[R3:\$[0-9]+]], [[R1]][1]
  ; ALL-NOT: sll
  ; ALL-NOT: sra

  ret i64 %3
}

define i32 @extract_zext_v16i8() nounwind {
  ; ALL-LABEL: extract_zext_v16i8:

  %1 = load <16 x i8>, <16 x i8>* @v16i8
  ; ALL-DAG: ld.b [[R1:\$w[0-9]+]],

  %2 = add <16 x i8> %1, %1
  ; ALL-DAG: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <16 x i8> %2, i32 1
  %4 = zext i8 %3 to i32
  ; ALL-DAG: copy_u.b [[R3:\$[0-9]+]], [[R1]][1]
  ; ALL-NOT: andi

  ret i32 %4
}

define i32 @extract_zext_v8i16() nounwind {
  ; ALL-LABEL: extract_zext_v8i16:

  %1 = load <8 x i16>, <8 x i16>* @v8i16
  ; ALL-DAG: ld.h [[R1:\$w[0-9]+]],

  %2 = add <8 x i16> %1, %1
  ; ALL-DAG: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <8 x i16> %2, i32 1
  %4 = zext i16 %3 to i32
  ; ALL-DAG: copy_u.h [[R3:\$[0-9]+]], [[R1]][1]
  ; ALL-NOT: andi

  ret i32 %4
}

define i32 @extract_zext_v4i32() nounwind {
  ; ALL-LABEL: extract_zext_v4i32:

  %1 = load <4 x i32>, <4 x i32>* @v4i32
  ; ALL-DAG: ld.w [[R1:\$w[0-9]+]],

  %2 = add <4 x i32> %1, %1
  ; ALL-DAG: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <4 x i32> %2, i32 1
  ; ALL-DAG: copy_{{[su]}}.w [[R3:\$[0-9]+]], [[R1]][1]

  ret i32 %3
}

define i64 @extract_zext_v2i64() nounwind {
  ; ALL-LABEL: extract_zext_v2i64:

  %1 = load <2 x i64>, <2 x i64>* @v2i64
  ; ALL-DAG: ld.d [[R1:\$w[0-9]+]],

  %2 = add <2 x i64> %1, %1
  ; ALL-DAG: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = extractelement <2 x i64> %2, i32 1
  ; MIPS32-DAG: copy_{{[su]}}.w [[R3:\$[0-9]+]], [[R1]][2]
  ; MIPS32-DAG: copy_{{[su]}}.w [[R4:\$[0-9]+]], [[R1]][3]
  ; MIPS64-DAG: copy_{{[su]}}.d [[R3:\$[0-9]+]], [[R1]][1]
  ; ALL-NOT: andi

  ret i64 %3
}

define i32 @extract_sext_v16i8_vidx() nounwind {
  ; ALL-LABEL: extract_sext_v16i8_vidx:

  %1 = load <16 x i8>, <16 x i8>* @v16i8
  ; O32-DAG: lw [[PTR_V:\$[0-9]+]], %got(v16i8)(
  ; N32-DAG: lw [[PTR_V:\$[0-9]+]], %got_disp(v16i8)(
  ; N64-DAG: ld [[PTR_V:\$[0-9]+]], %got_disp(v16i8)(
  ; ALL-DAG: ld.b [[R1:\$w[0-9]+]], 0([[PTR_V]])

  %2 = add <16 x i8> %1, %1
  ; ALL-DAG: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = load i32, i32* @i32
  ; O32-DAG: lw [[PTR_I:\$[0-9]+]], %got(i32)(
  ; N32-DAG: lw [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; N64-DAG: ld [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; ALL-DAG: lw [[IDX:\$[0-9]+]], 0([[PTR_I]])

  %4 = extractelement <16 x i8> %2, i32 %3
  %5 = sext i8 %4 to i32
  ; ALL-DAG: splat.b $w[[R3:[0-9]+]], [[R1]]{{\[}}[[IDX]]]
  ; ALL-DAG: mfc1 [[R5:\$[0-9]+]], $f[[R3]]
  ; ALL-DAG: sra [[R6:\$[0-9]+]], [[R5]], 24

  ret i32 %5
}

define i32 @extract_sext_v8i16_vidx() nounwind {
  ; ALL-LABEL: extract_sext_v8i16_vidx:

  %1 = load <8 x i16>, <8 x i16>* @v8i16
  ; O32-DAG: lw [[PTR_V:\$[0-9]+]], %got(v8i16)(
  ; N32-DAG: lw [[PTR_V:\$[0-9]+]], %got_disp(v8i16)(
  ; N64-DAG: ld [[PTR_V:\$[0-9]+]], %got_disp(v8i16)(
  ; ALL-DAG: ld.h [[R1:\$w[0-9]+]], 0([[PTR_V]])

  %2 = add <8 x i16> %1, %1
  ; ALL-DAG: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = load i32, i32* @i32
  ; O32-DAG: lw [[PTR_I:\$[0-9]+]], %got(i32)(
  ; N32-DAG: lw [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; N64-DAG: ld [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; ALL-DAG: lw [[IDX:\$[0-9]+]], 0([[PTR_I]])

  %4 = extractelement <8 x i16> %2, i32 %3
  %5 = sext i16 %4 to i32
  ; ALL-DAG: splat.h $w[[R3:[0-9]+]], [[R1]]{{\[}}[[IDX]]]
  ; ALL-DAG: mfc1 [[R5:\$[0-9]+]], $f[[R3]]
  ; ALL-DAG: sra [[R6:\$[0-9]+]], [[R5]], 16

  ret i32 %5
}

define i32 @extract_sext_v4i32_vidx() nounwind {
  ; ALL-LABEL: extract_sext_v4i32_vidx:

  %1 = load <4 x i32>, <4 x i32>* @v4i32
  ; O32-DAG: lw [[PTR_V:\$[0-9]+]], %got(v4i32)(
  ; N32-DAG: lw [[PTR_V:\$[0-9]+]], %got_disp(v4i32)(
  ; N64-DAG: ld [[PTR_V:\$[0-9]+]], %got_disp(v4i32)(
  ; ALL-DAG: ld.w [[R1:\$w[0-9]+]], 0([[PTR_V]])

  %2 = add <4 x i32> %1, %1
  ; ALL-DAG: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = load i32, i32* @i32
  ; O32-DAG: lw [[PTR_I:\$[0-9]+]], %got(i32)(
  ; N32-DAG: lw [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; N64-DAG: ld [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; ALL-DAG: lw [[IDX:\$[0-9]+]], 0([[PTR_I]])

  %4 = extractelement <4 x i32> %2, i32 %3
  ; ALL-DAG: splat.w $w[[R3:[0-9]+]], [[R1]]{{\[}}[[IDX]]]
  ; ALL-DAG: mfc1 [[R5:\$[0-9]+]], $f[[R3]]
  ; ALL-NOT: sra

  ret i32 %4
}

define i64 @extract_sext_v2i64_vidx() nounwind {
  ; ALL-LABEL: extract_sext_v2i64_vidx:

  %1 = load <2 x i64>, <2 x i64>* @v2i64
  ; O32-DAG: lw [[PTR_V:\$[0-9]+]], %got(v2i64)(
  ; N32-DAG: lw [[PTR_V:\$[0-9]+]], %got_disp(v2i64)(
  ; N64-DAG: ld [[PTR_V:\$[0-9]+]], %got_disp(v2i64)(
  ; ALL-DAG: ld.d [[R1:\$w[0-9]+]], 0([[PTR_V]])

  %2 = add <2 x i64> %1, %1
  ; ALL-DAG: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = load i32, i32* @i32
  ; O32-DAG: lw [[PTR_I:\$[0-9]+]], %got(i32)(
  ; N32-DAG: lw [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; N64-DAG: ld [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; ALL-DAG: lw [[IDX:\$[0-9]+]], 0([[PTR_I]])

  %4 = extractelement <2 x i64> %2, i32 %3
  ; MIPS32-DAG: splat.w $w[[R3:[0-9]+]], [[R1]]{{\[}}[[IDX]]]
  ; MIPS32-DAG: mfc1 [[R5:\$[0-9]+]], $f[[R3]]
  ; MIPS32-DAG: splat.w $w[[R4:[0-9]+]], [[R1]]{{\[}}[[IDX]]]
  ; MIPS32-DAG: mfc1 [[R6:\$[0-9]+]], $f[[R4]]
  ; MIPS64-DAG: splat.d $w[[R3:[0-9]+]], [[R1]]{{\[}}[[IDX]]]
  ; MIPS64-DAG: dmfc1 [[R5:\$[0-9]+]], $f[[R3]]
  ; ALL-NOT: sra

  ret i64 %4
}

define i32 @extract_zext_v16i8_vidx() nounwind {
  ; ALL-LABEL: extract_zext_v16i8_vidx:

  %1 = load <16 x i8>, <16 x i8>* @v16i8
  ; O32-DAG: lw [[PTR_V:\$[0-9]+]], %got(v16i8)(
  ; N32-DAG: lw [[PTR_V:\$[0-9]+]], %got_disp(v16i8)(
  ; N64-DAG: ld [[PTR_V:\$[0-9]+]], %got_disp(v16i8)(
  ; ALL-DAG: ld.b [[R1:\$w[0-9]+]], 0([[PTR_V]])

  %2 = add <16 x i8> %1, %1
  ; ALL-DAG: addv.b [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = load i32, i32* @i32
  ; O32-DAG: lw [[PTR_I:\$[0-9]+]], %got(i32)(
  ; N32-DAG: lw [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; N64-DAG: ld [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; ALL-DAG: lw [[IDX:\$[0-9]+]], 0([[PTR_I]])

  %4 = extractelement <16 x i8> %2, i32 %3
  %5 = zext i8 %4 to i32
  ; ALL-DAG: splat.b $w[[R3:[0-9]+]], [[R1]]{{\[}}[[IDX]]]
  ; ALL-DAG: mfc1 [[R5:\$[0-9]+]], $f[[R3]]
  ; ALL-DAG: srl [[R6:\$[0-9]+]], [[R5]], 24

  ret i32 %5
}

define i32 @extract_zext_v8i16_vidx() nounwind {
  ; ALL-LABEL: extract_zext_v8i16_vidx:

  %1 = load <8 x i16>, <8 x i16>* @v8i16
  ; O32-DAG: lw [[PTR_V:\$[0-9]+]], %got(v8i16)(
  ; N32-DAG: lw [[PTR_V:\$[0-9]+]], %got_disp(v8i16)(
  ; N64-DAG: ld [[PTR_V:\$[0-9]+]], %got_disp(v8i16)(
  ; ALL-DAG: ld.h [[R1:\$w[0-9]+]], 0([[PTR_V]])

  %2 = add <8 x i16> %1, %1
  ; ALL-DAG: addv.h [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = load i32, i32* @i32
  ; O32-DAG: lw [[PTR_I:\$[0-9]+]], %got(i32)(
  ; N32-DAG: lw [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; N64-DAG: ld [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; ALL-DAG: lw [[IDX:\$[0-9]+]], 0([[PTR_I]])

  %4 = extractelement <8 x i16> %2, i32 %3
  %5 = zext i16 %4 to i32
  ; ALL-DAG: splat.h $w[[R3:[0-9]+]], [[R1]]{{\[}}[[IDX]]]
  ; ALL-DAG: mfc1 [[R5:\$[0-9]+]], $f[[R3]]
  ; ALL-DAG: srl [[R6:\$[0-9]+]], [[R5]], 16

  ret i32 %5
}

define i32 @extract_zext_v4i32_vidx() nounwind {
  ; ALL-LABEL: extract_zext_v4i32_vidx:

  %1 = load <4 x i32>, <4 x i32>* @v4i32
  ; O32-DAG: lw [[PTR_V:\$[0-9]+]], %got(v4i32)(
  ; N32-DAG: lw [[PTR_V:\$[0-9]+]], %got_disp(v4i32)(
  ; N64-DAG: ld [[PTR_V:\$[0-9]+]], %got_disp(v4i32)(
  ; ALL-DAG: ld.w [[R1:\$w[0-9]+]], 0([[PTR_V]])

  %2 = add <4 x i32> %1, %1
  ; ALL-DAG: addv.w [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = load i32, i32* @i32
  ; O32-DAG: lw [[PTR_I:\$[0-9]+]], %got(i32)(
  ; N32-DAG: lw [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; N64-DAG: ld [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; ALL-DAG: lw [[IDX:\$[0-9]+]], 0([[PTR_I]])

  %4 = extractelement <4 x i32> %2, i32 %3
  ; ALL-DAG: splat.w $w[[R3:[0-9]+]], [[R1]]{{\[}}[[IDX]]]
  ; ALL-DAG: mfc1 [[R5:\$[0-9]+]], $f[[R3]]
  ; ALL-NOT: srl

  ret i32 %4
}

define i64 @extract_zext_v2i64_vidx() nounwind {
  ; ALL-LABEL: extract_zext_v2i64_vidx:

  %1 = load <2 x i64>, <2 x i64>* @v2i64
  ; O32-DAG: lw [[PTR_V:\$[0-9]+]], %got(v2i64)(
  ; N32-DAG: lw [[PTR_V:\$[0-9]+]], %got_disp(v2i64)(
  ; N64-DAG: ld [[PTR_V:\$[0-9]+]], %got_disp(v2i64)(
  ; ALL-DAG: ld.d [[R1:\$w[0-9]+]], 0([[PTR_V]])

  %2 = add <2 x i64> %1, %1
  ; ALL-DAG: addv.d [[R2:\$w[0-9]+]], [[R1]], [[R1]]

  %3 = load i32, i32* @i32
  ; O32-DAG: lw [[PTR_I:\$[0-9]+]], %got(i32)(
  ; N32-DAG: lw [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; N64-DAG: ld [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; ALL-DAG: lw [[IDX:\$[0-9]+]], 0([[PTR_I]])

  %4 = extractelement <2 x i64> %2, i32 %3
  ; MIPS32-DAG: splat.w $w[[R3:[0-9]+]], [[R1]]{{\[}}[[IDX]]]
  ; MIPS32-DAG: mfc1 [[R5:\$[0-9]+]], $f[[R3]]
  ; MIPS32-DAG: splat.w $w[[R4:[0-9]+]], [[R1]]{{\[}}[[IDX]]]
  ; MIPS32-DAG: mfc1 [[R6:\$[0-9]+]], $f[[R4]]
  ; MIPS64-DAG: splat.d $w[[R3:[0-9]+]], [[R1]]{{\[}}[[IDX]]]
  ; MIPS64-DAG: dmfc1 [[R5:\$[0-9]+]], $f[[R3]]
  ; ALL-NOT: srl

  ret i64 %4
}

define void @insert_v16i8(i32 signext %a) nounwind {
  ; ALL-LABEL: insert_v16i8:

  %1 = load <16 x i8>, <16 x i8>* @v16i8
  ; ALL-DAG: ld.b [[R1:\$w[0-9]+]],

  %a2 = trunc i32 %a to i8
  %a3 = sext i8 %a2 to i32
  %a4 = trunc i32 %a3 to i8
  ; ALL-NOT: andi
  ; ALL-NOT: sra

  %2 = insertelement <16 x i8> %1, i8 %a4, i32 1
  ; ALL-DAG: insert.b [[R1]][1], $4

  store <16 x i8> %2, <16 x i8>* @v16i8
  ; ALL-DAG: st.b [[R1]]

  ret void
}

define void @insert_v8i16(i32 signext %a) nounwind {
  ; ALL-LABEL: insert_v8i16:

  %1 = load <8 x i16>, <8 x i16>* @v8i16
  ; ALL-DAG: ld.h [[R1:\$w[0-9]+]],

  %a2 = trunc i32 %a to i16
  %a3 = sext i16 %a2 to i32
  %a4 = trunc i32 %a3 to i16
  ; ALL-NOT: andi
  ; ALL-NOT: sra

  %2 = insertelement <8 x i16> %1, i16 %a4, i32 1
  ; ALL-DAG: insert.h [[R1]][1], $4

  store <8 x i16> %2, <8 x i16>* @v8i16
  ; ALL-DAG: st.h [[R1]]

  ret void
}

define void @insert_v4i32(i32 signext %a) nounwind {
  ; ALL-LABEL: insert_v4i32:

  %1 = load <4 x i32>, <4 x i32>* @v4i32
  ; ALL-DAG: ld.w [[R1:\$w[0-9]+]],

  ; ALL-NOT: andi
  ; ALL-NOT: sra

  %2 = insertelement <4 x i32> %1, i32 %a, i32 1
  ; ALL-DAG: insert.w [[R1]][1], $4

  store <4 x i32> %2, <4 x i32>* @v4i32
  ; ALL-DAG: st.w [[R1]]

  ret void
}

define void @insert_v2i64(i64 signext %a) nounwind {
  ; ALL-LABEL: insert_v2i64:

  %1 = load <2 x i64>, <2 x i64>* @v2i64
  ; MIPS32-DAG: ld.w [[R1:\$w[0-9]+]],
  ; MIPS64-DAG: ld.d [[R1:\$w[0-9]+]],

  ; ALL-NOT: andi
  ; ALL-NOT: sra

  %2 = insertelement <2 x i64> %1, i64 %a, i32 1
  ; MIPS32-DAG: insert.w [[R1]][2], $4
  ; MIPS32-DAG: insert.w [[R1]][3], $5
  ; MIPS64-DAG: insert.d [[R1]][1], $4

  store <2 x i64> %2, <2 x i64>* @v2i64
  ; MIPS32-DAG: st.w [[R1]]
  ; MIPS64-DAG: st.d [[R1]]

  ret void
}

define void @insert_v16i8_vidx(i32 signext %a) nounwind {
  ; ALL-LABEL: insert_v16i8_vidx:

  %1 = load <16 x i8>, <16 x i8>* @v16i8
  ; ALL-DAG: ld.b [[R1:\$w[0-9]+]],

  %2 = load i32, i32* @i32
  ; O32-DAG: lw [[PTR_I:\$[0-9]+]], %got(i32)(
  ; N32-DAG: lw [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; N64-DAG: ld [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; ALL-DAG: lw [[IDX:\$[0-9]+]], 0([[PTR_I]])

  %a2 = trunc i32 %a to i8
  %a3 = sext i8 %a2 to i32
  %a4 = trunc i32 %a3 to i8
  ; ALL-NOT: andi
  ; ALL-NOT: sra

  %3 = insertelement <16 x i8> %1, i8 %a4, i32 %2
  ; ALL-DAG: sld.b [[R1]], [[R1]]{{\[}}[[IDX]]]
  ; ALL-DAG: insert.b [[R1]][0], $4
  ; O32-DAG: neg [[NIDX:\$[0-9]+]], [[IDX]]
  ; N32-DAG: neg [[NIDX:\$[0-9]+]], [[IDX]]
  ; N64-DAG: dneg [[NIDX:\$[0-9]+]], [[IDX]]
  ; ALL-DAG: sld.b [[R1]], [[R1]]{{\[}}[[NIDX]]]

  store <16 x i8> %3, <16 x i8>* @v16i8
  ; ALL-DAG: st.b [[R1]]

  ret void
}

define void @insert_v8i16_vidx(i32 signext %a) nounwind {
  ; ALL-LABEL: insert_v8i16_vidx:

  %1 = load <8 x i16>, <8 x i16>* @v8i16
  ; ALL-DAG: ld.h [[R1:\$w[0-9]+]],

  %2 = load i32, i32* @i32
  ; O32-DAG: lw [[PTR_I:\$[0-9]+]], %got(i32)(
  ; N32-DAG: lw [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; N64-DAG: ld [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; ALL-DAG: lw [[IDX:\$[0-9]+]], 0([[PTR_I]])

  %a2 = trunc i32 %a to i16
  %a3 = sext i16 %a2 to i32
  %a4 = trunc i32 %a3 to i16
  ; ALL-NOT: andi
  ; ALL-NOT: sra

  %3 = insertelement <8 x i16> %1, i16 %a4, i32 %2
  ; ALL-DAG: sll [[BIDX:\$[0-9]+]], [[IDX]], 1
  ; ALL-DAG: sld.b [[R1]], [[R1]]{{\[}}[[BIDX]]]
  ; ALL-DAG: insert.h [[R1]][0], $4
  ; O32-DAG: neg [[NIDX:\$[0-9]+]], [[BIDX]]
  ; N32-DAG: neg [[NIDX:\$[0-9]+]], [[BIDX]]
  ; N64-DAG: dneg [[NIDX:\$[0-9]+]], [[BIDX]]
  ; ALL-DAG: sld.b [[R1]], [[R1]]{{\[}}[[NIDX]]]

  store <8 x i16> %3, <8 x i16>* @v8i16
  ; ALL-DAG: st.h [[R1]]

  ret void
}

define void @insert_v4i32_vidx(i32 signext %a) nounwind {
  ; ALL-LABEL: insert_v4i32_vidx:

  %1 = load <4 x i32>, <4 x i32>* @v4i32
  ; ALL-DAG: ld.w [[R1:\$w[0-9]+]],

  %2 = load i32, i32* @i32
  ; O32-DAG: lw [[PTR_I:\$[0-9]+]], %got(i32)(
  ; N32-DAG: lw [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; N64-DAG: ld [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; ALL-DAG: lw [[IDX:\$[0-9]+]], 0([[PTR_I]])

  ; ALL-NOT: andi
  ; ALL-NOT: sra

  %3 = insertelement <4 x i32> %1, i32 %a, i32 %2
  ; ALL-DAG: sll [[BIDX:\$[0-9]+]], [[IDX]], 2
  ; ALL-DAG: sld.b [[R1]], [[R1]]{{\[}}[[BIDX]]]
  ; ALL-DAG: insert.w [[R1]][0], $4
  ; O32-DAG: neg [[NIDX:\$[0-9]+]], [[BIDX]]
  ; N32-DAG: neg [[NIDX:\$[0-9]+]], [[BIDX]]
  ; N64-DAG: dneg [[NIDX:\$[0-9]+]], [[BIDX]]
  ; ALL-DAG: sld.b [[R1]], [[R1]]{{\[}}[[NIDX]]]

  store <4 x i32> %3, <4 x i32>* @v4i32
  ; ALL-DAG: st.w [[R1]]

  ret void
}

define void @insert_v2i64_vidx(i64 signext %a) nounwind {
  ; ALL-LABEL: insert_v2i64_vidx:

  %1 = load <2 x i64>, <2 x i64>* @v2i64
  ; MIPS32-DAG: ld.w [[R1:\$w[0-9]+]],
  ; MIPS64-DAG: ld.d [[R1:\$w[0-9]+]],

  %2 = load i32, i32* @i32
  ; O32-DAG: lw [[PTR_I:\$[0-9]+]], %got(i32)(
  ; N32-DAG: lw [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; N64-DAG: ld [[PTR_I:\$[0-9]+]], %got_disp(i32)(
  ; ALL-DAG: lw [[IDX:\$[0-9]+]], 0([[PTR_I]])

  ; ALL-NOT: andi
  ; ALL-NOT: sra

  %3 = insertelement <2 x i64> %1, i64 %a, i32 %2
  ; TODO: This code could be a lot better but it works. The legalizer splits
  ; 64-bit inserts into two 32-bit inserts because there is no i64 type on
  ; MIPS32. The obvious optimisation is to perform both insert.w's at once while
  ; the vector is rotated.
  ; MIPS32-DAG: sll [[BIDX:\$[0-9]+]], [[IDX]], 2
  ; MIPS32-DAG: sld.b [[R1]], [[R1]]{{\[}}[[BIDX]]]
  ; MIPS32-DAG: insert.w [[R1]][0], $4
  ; MIPS32-DAG: neg [[NIDX:\$[0-9]+]], [[BIDX]]
  ; MIPS32-DAG: sld.b [[R1]], [[R1]]{{\[}}[[NIDX]]]
  ; MIPS32-DAG: addiu [[IDX2:\$[0-9]+]], [[IDX]], 1
  ; MIPS32-DAG: sll [[BIDX:\$[0-9]+]], [[IDX2]], 2
  ; MIPS32-DAG: sld.b [[R1]], [[R1]]{{\[}}[[BIDX]]]
  ; MIPS32-DAG: insert.w [[R1]][0], $5
  ; MIPS32-DAG: neg [[NIDX:\$[0-9]+]], [[BIDX]]
  ; MIPS32-DAG: sld.b [[R1]], [[R1]]{{\[}}[[NIDX]]]

  ; MIPS64-DAG: sll [[BIDX:\$[0-9]+]], [[IDX]], 3
  ; MIPS64-DAG: sld.b [[R1]], [[R1]]{{\[}}[[BIDX]]]
  ; MIPS64-DAG: insert.d [[R1]][0], $4
  ; N32-DAG: neg [[NIDX:\$[0-9]+]], [[BIDX]]
  ; N64-DAG: dneg [[NIDX:\$[0-9]+]], [[BIDX]]
  ; MIPS64-DAG: sld.b [[R1]], [[R1]]{{\[}}[[NIDX]]]

  store <2 x i64> %3, <2 x i64>* @v2i64
  ; MIPS32-DAG: st.w [[R1]]
  ; MIPS64-DAG: st.d [[R1]]

  ret void
}

define void @truncstore() nounwind {
  ; ALL-LABEL: truncstore:

  store volatile <4 x i8> <i8 -1, i8 -1, i8 -1, i8 -1>, <4 x i8>*@v4i8
  ; TODO: What code should be emitted?

  ret void
}
