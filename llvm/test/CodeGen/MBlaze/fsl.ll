; Ensure that the FSL instrinsic instruction generate single FSL instructions
; at the machine level. Additionally, ensure that dynamic values use the
; dynamic version of the instructions and that constant values use the
; constant version of the instructions.
;
; RUN: llc -O3 < %s -march=mblaze | FileCheck %s

declare i32 @llvm.mblaze.fsl.get(i32 %port)
declare i32 @llvm.mblaze.fsl.aget(i32 %port)
declare i32 @llvm.mblaze.fsl.cget(i32 %port)
declare i32 @llvm.mblaze.fsl.caget(i32 %port)
declare i32 @llvm.mblaze.fsl.eget(i32 %port)
declare i32 @llvm.mblaze.fsl.eaget(i32 %port)
declare i32 @llvm.mblaze.fsl.ecget(i32 %port)
declare i32 @llvm.mblaze.fsl.ecaget(i32 %port)
declare i32 @llvm.mblaze.fsl.nget(i32 %port)
declare i32 @llvm.mblaze.fsl.naget(i32 %port)
declare i32 @llvm.mblaze.fsl.ncget(i32 %port)
declare i32 @llvm.mblaze.fsl.ncaget(i32 %port)
declare i32 @llvm.mblaze.fsl.neget(i32 %port)
declare i32 @llvm.mblaze.fsl.neaget(i32 %port)
declare i32 @llvm.mblaze.fsl.necget(i32 %port)
declare i32 @llvm.mblaze.fsl.necaget(i32 %port)
declare i32 @llvm.mblaze.fsl.tget(i32 %port)
declare i32 @llvm.mblaze.fsl.taget(i32 %port)
declare i32 @llvm.mblaze.fsl.tcget(i32 %port)
declare i32 @llvm.mblaze.fsl.tcaget(i32 %port)
declare i32 @llvm.mblaze.fsl.teget(i32 %port)
declare i32 @llvm.mblaze.fsl.teaget(i32 %port)
declare i32 @llvm.mblaze.fsl.tecget(i32 %port)
declare i32 @llvm.mblaze.fsl.tecaget(i32 %port)
declare i32 @llvm.mblaze.fsl.tnget(i32 %port)
declare i32 @llvm.mblaze.fsl.tnaget(i32 %port)
declare i32 @llvm.mblaze.fsl.tncget(i32 %port)
declare i32 @llvm.mblaze.fsl.tncaget(i32 %port)
declare i32 @llvm.mblaze.fsl.tneget(i32 %port)
declare i32 @llvm.mblaze.fsl.tneaget(i32 %port)
declare i32 @llvm.mblaze.fsl.tnecget(i32 %port)
declare i32 @llvm.mblaze.fsl.tnecaget(i32 %port)

declare void @llvm.mblaze.fsl.put(i32 %value, i32 %port)
declare void @llvm.mblaze.fsl.aput(i32 %value, i32 %port)
declare void @llvm.mblaze.fsl.cput(i32 %value, i32 %port)
declare void @llvm.mblaze.fsl.caput(i32 %value, i32 %port)
declare void @llvm.mblaze.fsl.nput(i32 %value, i32 %port)
declare void @llvm.mblaze.fsl.naput(i32 %value, i32 %port)
declare void @llvm.mblaze.fsl.ncput(i32 %value, i32 %port)
declare void @llvm.mblaze.fsl.ncaput(i32 %value, i32 %port)
declare void @llvm.mblaze.fsl.tput(i32 %port)
declare void @llvm.mblaze.fsl.taput(i32 %port)
declare void @llvm.mblaze.fsl.tcput(i32 %port)
declare void @llvm.mblaze.fsl.tcaput(i32 %port)
declare void @llvm.mblaze.fsl.tnput(i32 %port)
declare void @llvm.mblaze.fsl.tnaput(i32 %port)
declare void @llvm.mblaze.fsl.tncput(i32 %port)
declare void @llvm.mblaze.fsl.tncaput(i32 %port)

define void @fsl_get(i32 %port) {
    ; CHECK:        fsl_get:
    %v0  = call i32 @llvm.mblaze.fsl.get(i32 %port)
    ; CHECK:        getd
    %v1  = call i32 @llvm.mblaze.fsl.aget(i32 %port)
    ; CHECK-NEXT:   agetd
    %v2  = call i32 @llvm.mblaze.fsl.cget(i32 %port)
    ; CHECK-NEXT:   cgetd
    %v3  = call i32 @llvm.mblaze.fsl.caget(i32 %port)
    ; CHECK-NEXT:   cagetd
    %v4  = call i32 @llvm.mblaze.fsl.eget(i32 %port)
    ; CHECK-NEXT:   egetd
    %v5  = call i32 @llvm.mblaze.fsl.eaget(i32 %port)
    ; CHECK-NEXT:   eagetd
    %v6  = call i32 @llvm.mblaze.fsl.ecget(i32 %port)
    ; CHECK-NEXT:   ecgetd
    %v7  = call i32 @llvm.mblaze.fsl.ecaget(i32 %port)
    ; CHECK-NEXT:   ecagetd
    %v8  = call i32 @llvm.mblaze.fsl.nget(i32 %port)
    ; CHECK-NEXT:   ngetd
    %v9  = call i32 @llvm.mblaze.fsl.naget(i32 %port)
    ; CHECK-NEXT:   nagetd
    %v10 = call i32 @llvm.mblaze.fsl.ncget(i32 %port)
    ; CHECK-NEXT:   ncgetd
    %v11 = call i32 @llvm.mblaze.fsl.ncaget(i32 %port)
    ; CHECK-NEXT:   ncagetd
    %v12 = call i32 @llvm.mblaze.fsl.neget(i32 %port)
    ; CHECK-NEXT:   negetd
    %v13 = call i32 @llvm.mblaze.fsl.neaget(i32 %port)
    ; CHECK-NEXT:   neagetd
    %v14 = call i32 @llvm.mblaze.fsl.necget(i32 %port)
    ; CHECK-NEXT:   necgetd
    %v15 = call i32 @llvm.mblaze.fsl.necaget(i32 %port)
    ; CHECK-NEXT:   necagetd
    %v16 = call i32 @llvm.mblaze.fsl.tget(i32 %port)
    ; CHECK-NEXT:   tgetd
    %v17 = call i32 @llvm.mblaze.fsl.taget(i32 %port)
    ; CHECK-NEXT:   tagetd
    %v18 = call i32 @llvm.mblaze.fsl.tcget(i32 %port)
    ; CHECK-NEXT:   tcgetd
    %v19 = call i32 @llvm.mblaze.fsl.tcaget(i32 %port)
    ; CHECK-NEXT:   tcagetd
    %v20 = call i32 @llvm.mblaze.fsl.teget(i32 %port)
    ; CHECK-NEXT:   tegetd
    %v21 = call i32 @llvm.mblaze.fsl.teaget(i32 %port)
    ; CHECK-NEXT:   teagetd
    %v22 = call i32 @llvm.mblaze.fsl.tecget(i32 %port)
    ; CHECK-NEXT:   tecgetd
    %v23 = call i32 @llvm.mblaze.fsl.tecaget(i32 %port)
    ; CHECK-NEXT:   tecagetd
    %v24 = call i32 @llvm.mblaze.fsl.tnget(i32 %port)
    ; CHECK-NEXT:   tngetd
    %v25 = call i32 @llvm.mblaze.fsl.tnaget(i32 %port)
    ; CHECK-NEXT:   tnagetd
    %v26 = call i32 @llvm.mblaze.fsl.tncget(i32 %port)
    ; CHECK-NEXT:   tncgetd
    %v27 = call i32 @llvm.mblaze.fsl.tncaget(i32 %port)
    ; CHECK-NEXT:   tncagetd
    %v28 = call i32 @llvm.mblaze.fsl.tneget(i32 %port)
    ; CHECK-NEXT:   tnegetd
    %v29 = call i32 @llvm.mblaze.fsl.tneaget(i32 %port)
    ; CHECK-NEXT:   tneagetd
    %v30 = call i32 @llvm.mblaze.fsl.tnecget(i32 %port)
    ; CHECK-NEXT:   tnecgetd
    %v31 = call i32 @llvm.mblaze.fsl.tnecaget(i32 %port)
    ; CHECK-NEXT:   tnecagetd
    ret void
    ; CHECK:        rtsd
}

define void @fslc_get() {
    ; CHECK:        fslc_get:
    %v0  = call i32 @llvm.mblaze.fsl.get(i32 1)
    ; CHECK:        get
    %v1  = call i32 @llvm.mblaze.fsl.aget(i32 1)
    ; CHECK-NOT:    agetd
    ; CHECK:        aget
    %v2  = call i32 @llvm.mblaze.fsl.cget(i32 1)
    ; CHECK-NOT:    cgetd
    ; CHECK:        cget
    %v3  = call i32 @llvm.mblaze.fsl.caget(i32 1)
    ; CHECK-NOT:    cagetd
    ; CHECK:        caget
    %v4  = call i32 @llvm.mblaze.fsl.eget(i32 1)
    ; CHECK-NOT:    egetd
    ; CHECK:        eget
    %v5  = call i32 @llvm.mblaze.fsl.eaget(i32 1)
    ; CHECK-NOT:    eagetd
    ; CHECK:        eaget
    %v6  = call i32 @llvm.mblaze.fsl.ecget(i32 1)
    ; CHECK-NOT:    ecgetd
    ; CHECK:        ecget
    %v7  = call i32 @llvm.mblaze.fsl.ecaget(i32 1)
    ; CHECK-NOT:    ecagetd
    ; CHECK:        ecaget
    %v8  = call i32 @llvm.mblaze.fsl.nget(i32 1)
    ; CHECK-NOT:    ngetd
    ; CHECK:        nget
    %v9  = call i32 @llvm.mblaze.fsl.naget(i32 1)
    ; CHECK-NOT:    nagetd
    ; CHECK:        naget
    %v10 = call i32 @llvm.mblaze.fsl.ncget(i32 1)
    ; CHECK-NOT:    ncgetd
    ; CHECK:        ncget
    %v11 = call i32 @llvm.mblaze.fsl.ncaget(i32 1)
    ; CHECK-NOT:    ncagetd
    ; CHECK:        ncaget
    %v12 = call i32 @llvm.mblaze.fsl.neget(i32 1)
    ; CHECK-NOT:    negetd
    ; CHECK:        neget
    %v13 = call i32 @llvm.mblaze.fsl.neaget(i32 1)
    ; CHECK-NOT:    neagetd
    ; CHECK:        neaget
    %v14 = call i32 @llvm.mblaze.fsl.necget(i32 1)
    ; CHECK-NOT:    necgetd
    ; CHECK:        necget
    %v15 = call i32 @llvm.mblaze.fsl.necaget(i32 1)
    ; CHECK-NOT:    necagetd
    ; CHECK:        necaget
    %v16 = call i32 @llvm.mblaze.fsl.tget(i32 1)
    ; CHECK-NOT:    tgetd
    ; CHECK:        tget
    %v17 = call i32 @llvm.mblaze.fsl.taget(i32 1)
    ; CHECK-NOT:    tagetd
    ; CHECK:        taget
    %v18 = call i32 @llvm.mblaze.fsl.tcget(i32 1)
    ; CHECK-NOT:    tcgetd
    ; CHECK:        tcget
    %v19 = call i32 @llvm.mblaze.fsl.tcaget(i32 1)
    ; CHECK-NOT:    tcagetd
    ; CHECK:        tcaget
    %v20 = call i32 @llvm.mblaze.fsl.teget(i32 1)
    ; CHECK-NOT:    tegetd
    ; CHECK:        teget
    %v21 = call i32 @llvm.mblaze.fsl.teaget(i32 1)
    ; CHECK-NOT:    teagetd
    ; CHECK:        teaget
    %v22 = call i32 @llvm.mblaze.fsl.tecget(i32 1)
    ; CHECK-NOT:    tecgetd
    ; CHECK:        tecget
    %v23 = call i32 @llvm.mblaze.fsl.tecaget(i32 1)
    ; CHECK-NOT:    tecagetd
    ; CHECK:        tecaget
    %v24 = call i32 @llvm.mblaze.fsl.tnget(i32 1)
    ; CHECK-NOT:    tngetd
    ; CHECK:        tnget
    %v25 = call i32 @llvm.mblaze.fsl.tnaget(i32 1)
    ; CHECK-NOT:    tnagetd
    ; CHECK:        tnaget
    %v26 = call i32 @llvm.mblaze.fsl.tncget(i32 1)
    ; CHECK-NOT:    tncgetd
    ; CHECK:        tncget
    %v27 = call i32 @llvm.mblaze.fsl.tncaget(i32 1)
    ; CHECK-NOT:    tncagetd
    ; CHECK:        tncaget
    %v28 = call i32 @llvm.mblaze.fsl.tneget(i32 1)
    ; CHECK-NOT:    tnegetd
    ; CHECK:        tneget
    %v29 = call i32 @llvm.mblaze.fsl.tneaget(i32 1)
    ; CHECK-NOT:    tneagetd
    ; CHECK:        tneaget
    %v30 = call i32 @llvm.mblaze.fsl.tnecget(i32 1)
    ; CHECK-NOT:    tnecgetd
    ; CHECK:        tnecget
    %v31 = call i32 @llvm.mblaze.fsl.tnecaget(i32 1)
    ; CHECK-NOT:    tnecagetd
    ; CHECK:        tnecaget
    ret void
    ; CHECK:        rtsd
}

define void @putfsl(i32 %value, i32 %port) {
    ; CHECK:        putfsl:
    call void @llvm.mblaze.fsl.put(i32 %value, i32 %port)
    ; CHECK:        putd
    call void @llvm.mblaze.fsl.aput(i32 %value, i32 %port)
    ; CHECK-NEXT:   aputd
    call void @llvm.mblaze.fsl.cput(i32 %value, i32 %port)
    ; CHECK-NEXT:   cputd
    call void @llvm.mblaze.fsl.caput(i32 %value, i32 %port)
    ; CHECK-NEXT:   caputd
    call void @llvm.mblaze.fsl.nput(i32 %value, i32 %port)
    ; CHECK-NEXT:   nputd
    call void @llvm.mblaze.fsl.naput(i32 %value, i32 %port)
    ; CHECK-NEXT:   naputd
    call void @llvm.mblaze.fsl.ncput(i32 %value, i32 %port)
    ; CHECK-NEXT:   ncputd
    call void @llvm.mblaze.fsl.ncaput(i32 %value, i32 %port)
    ; CHECK-NEXT:   ncaputd
    call void @llvm.mblaze.fsl.tput(i32 %port)
    ; CHECK-NEXT:   tputd
    call void @llvm.mblaze.fsl.taput(i32 %port)
    ; CHECK-NEXT:   taputd
    call void @llvm.mblaze.fsl.tcput(i32 %port)
    ; CHECK-NEXT:   tcputd
    call void @llvm.mblaze.fsl.tcaput(i32 %port)
    ; CHECK-NEXT:   tcaputd
    call void @llvm.mblaze.fsl.tnput(i32 %port)
    ; CHECK-NEXT:   tnputd
    call void @llvm.mblaze.fsl.tnaput(i32 %port)
    ; CHECK-NEXT:   tnaputd
    call void @llvm.mblaze.fsl.tncput(i32 %port)
    ; CHECK-NEXT:   tncputd
    call void @llvm.mblaze.fsl.tncaput(i32 %port)
    ; CHECK-NEXT:   tncaputd
    ret void
    ; CHECK:        rtsd
}

define void @putfsl_const(i32 %value) {
    ; CHECK:        putfsl_const:
    call void @llvm.mblaze.fsl.put(i32 %value, i32 1)
    ; CHECK-NOT:    putd
    ; CHECK:        put
    call void @llvm.mblaze.fsl.aput(i32 %value, i32 1)
    ; CHECK-NOT:    aputd
    ; CHECK:        aput
    call void @llvm.mblaze.fsl.cput(i32 %value, i32 1)
    ; CHECK-NOT:    cputd
    ; CHECK:        cput
    call void @llvm.mblaze.fsl.caput(i32 %value, i32 1)
    ; CHECK-NOT:    caputd
    ; CHECK:        caput
    call void @llvm.mblaze.fsl.nput(i32 %value, i32 1)
    ; CHECK-NOT:    nputd
    ; CHECK:        nput
    call void @llvm.mblaze.fsl.naput(i32 %value, i32 1)
    ; CHECK-NOT:    naputd
    ; CHECK:        naput
    call void @llvm.mblaze.fsl.ncput(i32 %value, i32 1)
    ; CHECK-NOT:    ncputd
    ; CHECK:        ncput
    call void @llvm.mblaze.fsl.ncaput(i32 %value, i32 1)
    ; CHECK-NOT:    ncaputd
    ; CHECK:        ncaput
    call void @llvm.mblaze.fsl.tput(i32 1)
    ; CHECK-NOT:    tputd
    ; CHECK:        tput
    call void @llvm.mblaze.fsl.taput(i32 1)
    ; CHECK-NOT:    taputd
    ; CHECK:        taput
    call void @llvm.mblaze.fsl.tcput(i32 1)
    ; CHECK-NOT:    tcputd
    ; CHECK:        tcput
    call void @llvm.mblaze.fsl.tcaput(i32 1)
    ; CHECK-NOT:    tcaputd
    ; CHECK:        tcaput
    call void @llvm.mblaze.fsl.tnput(i32 1)
    ; CHECK-NOT:    tnputd
    ; CHECK:        tnput
    call void @llvm.mblaze.fsl.tnaput(i32 1)
    ; CHECK-NOT:    tnaputd
    ; CHECK:        tnaput
    call void @llvm.mblaze.fsl.tncput(i32 1)
    ; CHECK-NOT:    tncputd
    ; CHECK:        tncput
    call void @llvm.mblaze.fsl.tncaput(i32 1)
    ; CHECK-NOT:    tncaputd
    ; CHECK:        tncaput
    ret void
    ; CHECK:        rtsd
}
