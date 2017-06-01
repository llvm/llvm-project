; RUN: llc -march=hexagon < %s | FileCheck %s

@i65_l = external global i65
@i65_s = external global i65
@i129_l = external global i129
@i129_s = external global i129

; CHECK-LABEL: i129_ls
; CHECK-DAG: r[[REG0:[0-9:]+]] = memd(##i129_l)
; CHECK-DAG: r[[REG1:[0-9:]+]] = memd(##i129_l+8)
; CHECK-DAG: r[[REG2:[0-9]+]] = memub(##i129_l+16)
; CHECK-DAG: memb(##i129_s+16) = r[[REG2]]
; CHECK-DAG: memd(##i129_s+8) = r[[REG1]]
; CHECK-DAG: memd(##i129_s) = r[[REG0]]
define void @i129_ls() nounwind  {
        %tmp = load i129, i129* @i129_l
        store i129 %tmp, i129* @i129_s
        ret void
}

; CHECK-LABEL: i65_ls
; CHECK-DAG: r[[REG0:[0-9:]+]] = memd(##i65_l)
; CHECK-DAG: r[[REG1:[0-9]+]] = memub(##i65_l+8)
; CHECK-DAG: memd(##i65_s) = r[[REG0]]
; CHECK-DAG: memb(##i65_s+8) = r[[REG1]]
define void @i65_ls() nounwind  {
        %tmp = load i65, i65* @i65_l
        store i65 %tmp, i65* @i65_s
        ret void
}
