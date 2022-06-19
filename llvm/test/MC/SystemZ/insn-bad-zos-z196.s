* For z196 and above.
* RUN: not llvm-mc -triple s390x-ibm-zos -mcpu=z196 < %s 2> %t
* RUN: FileCheck < %t %s
* RUN: not llvm-mc -triple s390x-ibm-zos -mcpu=arch9 < %s 2> %t
* RUN: FileCheck < %t %s

*CHECK-COUNT-6: error: invalid instruction
        locrhe   1,2
        locgrhe  1,2
        loche    1,2(3)
        locghe   1,2(3)
        stoche   1,2(3)
        stocghe  1,2(3)


*CHECK-COUNT-6: error: invalid instruction
        locrle   1,2
        locgrle  1,2
        locle    1,2(3)
        locgle   1,2(3)
        stocle   1,2(3)
        stocgle  1,2(3)


*CHECK-COUNT-6: error: invalid instruction
        locrlh   1,2
        locgrlh  1,2
        loclh    1,2(3)
        locglh   1,2(3)
        stoclh   1,2(3)
        stocglh  1,2(3)


*CHECK-COUNT-6: error: invalid instruction
        locrnhe   1,2
        locgrnhe  1,2
        locnhe    1,2(3)
        locgnhe   1,2(3)
        stocnhe   1,2(3)
        stocgnhe  1,2(3)


*CHECK-COUNT-6: error: invalid instruction
        locrnle   1,2
        locgrnle  1,2
        locnle    1,2(3)
        locgnle   1,2(3)
        stocnle   1,2(3)
        stocgnle  1,2(3)


*CHECK-COUNT-6: error: invalid instruction
        locrnlh   1,2
        locgrnlh  1,2
        locnlh    1,2(3)
        locgnlh   1,2(3)
        stocnlh   1,2(3)
        stocgnlh  1,2(3)

