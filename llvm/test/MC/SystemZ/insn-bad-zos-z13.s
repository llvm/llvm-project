*For z13 only.
*RUN: not llvm-mc -triple s390x-ibm-zos -mcpu=z13 < %s 2> %t
*RUN: FileCheck < %t %s
*RUN: not llvm-mc -triple s390x-ibm-zos -mcpu=arch11 < %s 2> %t
*RUN: FileCheck < %t %s


*CHECK-COUNT-6: error: invalid instruction
        lochihe   1,2
        locghihe  1,2
        lochhihe  1,2
        locfhrhe  1,3
        locfhhe   1,2(3)
        stocfhhe  1,2(3)


*CHECK-COUNT-6: error: invalid instruction
        lochile   1,2
        locghile  1,2
        lochhile  1,2
        locfhrle  1,3
        locfhle   1,2(3)
        stocfhle  1,2(3)


*CHECK-COUNT-6: error: invalid instruction
        lochilh   1,2
        locghilh  1,2
        lochhilh  1,2
        locfhrlh  1,3
        locfhlh   1,2(3)
        stocfhlh  1,2(3)


*CHECK-COUNT-6: error: invalid instruction
        lochinhe   1,2
        locghinhe  1,2
        lochhinhe  1,2
        locfhrnhe  1,3
        locfhnhe   1,2(3)
        stocfhnhe  1,2(3)


*CHECK-COUNT-6: error: invalid instruction
        lochinle   1,2
        locghinle  1,2
        lochhinle  1,2
        locfhrnle  1,3
        locfhnle   1,2(3)
        stocfhnle  1,2(3)


*CHECK-COUNT-6: error: invalid instruction
        lochinlh   1,2
        locghinlh  1,2
        lochhinlh  1,2
        locfhrnlh  1,3
        locfhnlh  1,2(3)
        stocfhnlh  1,2(3)

