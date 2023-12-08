* For z10 only.
* RUN: not llvm-mc -triple s390x-ibm-zos -mcpu=z10 < %s 2> %t
* RUN: FileCheck < %t %s
* RUN: not llvm-mc -triple s390x-ibm-zos -mcpu=arch8 < %s 2> %t
* RUN: FileCheck < %t %s

*CHECK: error: invalid instruction
        jgnop       foo

*CHECK: error: invalid instruction
        jg          foo

*CHECK-COUNT-22: error: invalid instruction
        jnle        foo
        brnle       foo
        jgnle       foo
        brnlel      foo
        bnle        0(1)
        bnler       1
        crjnle      1,2,*+100
        cgrjnle     1,2,*+100
        cijnle      1,100,*+200
        cgijnle     1,100,*+200
        clrjnle     1,2,*+200
        clgrjnle    1,2,*+200
        clijnle     1,100,*+100
        clgijnle    1,100,*+100
        crtnle      0,foo
        cgrtnle     0,foo
        clrtnle     0,foo
        clgrtnle    0,foo
        citnle      1,foo
        cgitnle     1,foo
        clfitnle    1,foo
        clgitnle    1,foo


*CHECK-COUNT-22: error: invalid instruction
        jnhe        foo
        brnhe       foo
        jgnhe       foo
        brnhel      foo
        bnhe        0(1)
        bnher       1
        crjnhe      1,2,*+100
        cgrjnhe     1,2,*+100
        cijnhe      1,100,*+200
        clgijnle    1,100,*+100
        cgijnhe     1,100,*+200
        clrjnhe     1,2,*+200
        clgrjnhe    1,2,*+200
        clijnhe     1,100,*+100
        crtnhe      0,1
        cgrtnhe     0,1
        clrtnhe     0,1
        clgrtnhe    0,1
        citnhe      1,1
        cgitnhe     1,1
        clfitnhe    1,1
        clgitnhe    1,1


*CHECK-COUNT-22: error: invalid instruction
        jnlh        foo
        brnlh       foo
        jgnlh       foo
        brnlhl      foo
        bnlh        0(1)
        bnlhr       1
        crjnlh      1,2,*+200
        cgrjnlh     1,2,*+200
        cijnlh      1,100,*+200
        cgijnlh     1,100,*+200
        clrjnlh     1,2,*+200
        clgrjnlh    1,2,*+200
        clijnlh     1,100,*+100
        clgijnlh    1,100,*+100
        crtnlh      0,1
        cgrtnlh     0,1
        clrtnlh     0,1
        clgrtnlh    0,1
        citnlh      1,1
        cgitnlh     1,1
        clfitnlh    1,1
        clgitnlh    1,1

*CHECK-COUNT-22: error: invalid instruction
        brlh        foo
        jglh        foo
        jllh        foo
        brlhl       foo
        blh         0(1)
        blhr        1
        crjlh       1,2,*+100
        cgrjlh      1,2,*+100
        cijlh       1,100,*+200
        cgijlh      1,100,*+200
        clrjlh      1,2,*+200
        clgrjlh     1,2,*+200
        clijlh      1,100,*+100
        clgijlh     1,100,*+100
        crtlh       0,1
        cgrtlh      0,1
        clrtlh      0,1
        clgrtlh     0,1
        citlh       1,1
        cgitlh      1,1
        clfitlh     1,1
        clgitlh     1,1

*CHECK-COUNT-22: error: invalid instruction
        jhe         foo
        brhe        foo
        jghe        foo
        brhel       foo
        bhe         0(1)
        bher        1
        crjhe       1,2,*+200
        cgrjhe      1,2,*+200
        cijhe       1,100,*+200
        cgijhe      1,100,*+200
        clrjhe      1,2,*+200
        clgrjhe     1,2,*+200
        clijhe      1,100,*+100
        clgijhe     1,100,*+100
        crthe       0,1
        cgrthe      0,1
        clrthe      0,1
        clgrthe     0,1
        cithe       1,1
        cgithe      1,1
        clfithe     1,1
        clgithe     1,1

*CHECK-COUNT-22: error: invalid instruction
        brle        foo
        jgle        foo
        jlle        foo
        brlel       foo
        ble         0(1)
        bler        1
        crjle       1,2,*+200
        cgrjle      1,2,*+200
        cijle       1,100,*+200
        cgijle      1,100,*+200
        clrjle      1,2,*+200
        clgrjle     1,2,*+200
        clijle      1,100,*+200
        clgijle     1,100,*+100
        crtle       0,1
        cgrtle      0,1
        clrtle      0,1
        clgrtle     0,1
        citle       1,1
        cgitle      1,1
        clfitle     1,1
        clgitle     1,1

