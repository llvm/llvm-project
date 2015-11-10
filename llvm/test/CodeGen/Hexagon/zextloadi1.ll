; RUN: llc -march=hexagon -mcpu=hexagonv4 < %s | FileCheck %s

; CHECK: r{{[0-9]+}} = ##i129_l+16
; CHECK: r{{[0-9]+}} = ##i129_s+16
; CHECK: memd(##i129_s) = r{{[0-9]+:[0-9]+}}
; CHECK: r{{[0-9]+}} = ##i65_l+8
; CHECK: r{{[0-9]+}} = ##i65_s+8
; CHECK: memd(##i65_s) = r{{[0-9]+:[0-9]+}}

@i65_l = external global i65
@i65_s = external global i65
@i129_l = external global i129
@i129_s = external global i129

define void @i129_ls() nounwind  {
        %tmp = load i129, i129* @i129_l
        store i129 %tmp, i129* @i129_s
        ret void
}

define void @i65_ls() nounwind  {
        %tmp = load i65, i65* @i65_l
        store i65 %tmp, i65* @i65_s
        ret void
}
