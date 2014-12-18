; RUN: llc -mtriple=mips-unknown-linux-gnu -relocation-model=static < %s | FileCheck --check-prefix=ALL --check-prefix=SYM32 --check-prefix=O32-BE %s
; RUN: llc -mtriple=mipsel-unknown-linux-gnu -relocation-model=static < %s | FileCheck --check-prefix=ALL --check-prefix=SYM32 --check-prefix=O32-LE %s

; RUN-TODO: llc -mtriple=mips64-unknown-linux-gnu -relocation-model=static -mattr=-n64,+o32 < %s | FileCheck --check-prefix=ALL --check-prefix=SYM32 --check-prefix=O32-BE %s
; RUN-TODO: llc -mtriple=mips64el-unknown-linux-gnu -relocation-model=static -mattr=-n64,+o32 < %s | FileCheck --check-prefix=ALL --check-prefix=SYM32 --check-prefix=O32-LE %s

; RUN: llc -mtriple=mips64-unknown-linux-gnu -relocation-model=static -mattr=-n64,+n32 < %s | FileCheck --check-prefix=ALL --check-prefix=SYM32 --check-prefix=NEW-BE %s
; RUN: llc -mtriple=mips64el-unknown-linux-gnu -relocation-model=static -mattr=-n64,+n32 < %s | FileCheck --check-prefix=ALL --check-prefix=SYM32 --check-prefix=NEW-LE %s

; RUN: llc -mtriple=mips64-unknown-linux-gnu -relocation-model=static -mattr=-n64,+n64 < %s | FileCheck --check-prefix=ALL --check-prefix=SYM64 --check-prefix=NEW-BE %s
; RUN: llc -mtriple=mips64el-unknown-linux-gnu -relocation-model=static -mattr=-n64,+n64 < %s | FileCheck --check-prefix=ALL --check-prefix=SYM64 --check-prefix=NEW-LE %s

; Test small structures for all ABI's and byte orders.
;
; N32/N64 are identical in this area so their checks have been combined into
; the 'NEW' prefix (the N stands for New).

@bytes = global [2 x i8] zeroinitializer

define void @s_i8(i8 inreg %a) nounwind {
entry:
	store i8 %a, i8* getelementptr inbounds ([2 x i8]* @bytes, i32 0, i32 1)
        ret void
}

; ALL-LABEL: s_i8:

; SYM32-DAG:   lui   [[PTR_HI:\$[0-9]+]], %hi(bytes)
; SYM32-DAG:   addiu [[PTR:\$[0-9]+]], [[PTR_HI]], %lo(bytes)

; SYM64-DAG:   ld    [[PTR:\$[0-9]+]], %got_disp(bytes)(

; O32-BE-DAG:  srl [[ARG:\$[0-9]+]], $4, 24
; O32-BE-DAG:  sb  [[ARG]], 1([[PTR]])

; O32-LE-DAG:  sb  $4, 1([[PTR]])

; NEW-BE-DAG:  dsrl [[ARG:\$[0-9]+]], $4, 56
; NEW-BE-DAG:  sb   [[ARG]], 1([[PTR]])

; NEW-LE-DAG:  sb   $4, 1([[PTR]])
