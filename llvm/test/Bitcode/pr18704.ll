; RUN:  not llvm-dis < %s.bc 2>&1 | FileCheck %s

; CHECK: llvm-dis{{(\.EXE|\.exe)?}}: error: Invalid record

; pr18704.ll.bc has an instruction referring to invalid type.
; The test checks that LLVM reports the error and doesn't access freed memory
; in doing so.

;<MODULE_BLOCK NumWords=217 BlockCodeSize=3>
;  <VERSION op0=1/>
;  <BLOCKINFO_BLOCK/>
;  <TYPE_BLOCK_ID NumWords=23 BlockCodeSize=4>
;    <NUMENTRY op0=25/>
;    <INTEGER op0=8/>
;    <POINTER abbrevid=4 op0=0 op1=0/>
;    <POINTER abbrevid=4 op0=1 op1=0/>
;    <ARRAY abbrevid=9 op0=6 op1=0/>
;    <POINTER abbrevid=4 op0=3 op1=0/>
;    <ARRAY abbrevid=9 op0=10 op1=0/>
;    <POINTER abbrevid=4 op0=5 op1=0/>
;    <ARRAY abbrevid=9 op0=4 op1=0/>
;    <POINTER abbrevid=4 op0=7 op1=0/>
;    <ARRAY abbrevid=9 op0=5 op1=0/>
;    <POINTER abbrevid=4 op0=9 op1=0/>
;    <STRUCT_NAME abbrevid=7 op0=115 op1=116 op2=114 op3=117 op4=99 op5=116 op6=46 op7=112 op8=97 op9=105 op10=114 op11=46 op12=48/>
;    <STRUCT_NAMED abbrevid=8 op0=0 op1=1 op2=1/>
;    <ARRAY abbrevid=9 op0=2 op1=11/>
;    <POINTER abbrevid=4 op0=12 op1=0/>
;    <FUNCTION abbrevid=5 op0=0 op1=1 op2=1 op3=1/>
;    <POINTER abbrevid=4 op0=14 op1=0/>
;    <FUNCTION abbrevid=5 op0=0 op1=1 op2=1/>
;    <POINTER abbrevid=4 op0=16 op1=0/>
;    <INTEGER op0=64/>
;    <FUNCTION abbrevid=5 op0=0 op1=1 op2=18/>
;    <POINTER abbrevid=4 op0=19 op1=0/>
;    <INTEGER op0=32/>
;    <FUNCTION abbrevid=5 op0=0 op1=21/>
;    <POINTER abbrevid=4 op0=22 op1=0/>
;    <VOID/>
;  </TYPE_BLOCK_ID>
;  <GLOBALVAR abbrevid=4 op0=2 op1=0 op2=0 op3=0 op4=0 op5=0/>
;  <GLOBALVAR abbrevid=4 op0=2 op1=0 op2=0 op3=0 op4=0 op5=0/>
;  <GLOBALVAR abbrevid=4 op0=2 op1=0 op2=0 op3=0 op4=0 op5=0/>
;  <GLOBALVAR op0=4 op1=1 op2=25 op3=9 op4=0 op5=0 op6=0 op7=0 op8=1 op9=0/>
;  <GLOBALVAR op0=6 op1=1 op2=26 op3=9 op4=0 op5=0 op6=0 op7=0 op8=1 op9=0/>
;  <GLOBALVAR op0=8 op1=1 op2=27 op3=9 op4=0 op5=0 op6=0 op7=0 op8=1 op9=0/>
;  <GLOBALVAR abbrevid=4 op0=10 op1=1 op2=28 op3=3 op4=0 op5=0/>
;  <GLOBALVAR abbrevid=4 op0=6 op1=1 op2=26 op3=3 op4=0 op5=0/>
;  <GLOBALVAR abbrevid=4 op0=13 op1=1 op2=31 op3=3 op4=0 op5=0/>
;  <GLOBALVAR abbrevid=4 op0=2 op1=1 op2=23 op3=3 op4=0 op5=0/>
;  <GLOBALVAR abbrevid=4 op0=2 op1=0 op2=24 op3=0 op4=0 op5=0/>
;  <GLOBALVAR op0=10 op1=1 op2=28 op3=9 op4=0 op5=0 op6=0 op7=0 op8=1 op9=0/>
;  <FUNCTION op0=15 op1=0 op2=1 op3=0 op4=0 op5=0 op6=0 op7=0 op8=0 op9=0/>
;  <FUNCTION op0=17 op1=0 op2=1 op3=0 op4=0 op5=0 op6=0 op7=0 op8=0 op9=0/>
;  <FUNCTION op0=20 op1=0 op2=1 op3=0 op4=0 op5=0 op6=0 op7=0 op8=0 op9=0/>
;  <FUNCTION op0=15 op1=0 op2=0 op3=0 op4=0 op5=0 op6=0 op7=0 op8=0 op9=0/>
;  <FUNCTION op0=17 op1=0 op2=0 op3=0 op4=0 op5=0 op6=0 op7=0 op8=0 op9=0/>
;  <FUNCTION op0=23 op1=0 op2=0 op3=0 op4=0 op5=0 op6=0 op7=0 op8=0 op9=0/>
;  <CONSTANTS_BLOCK NumWords=20 BlockCodeSize=4>
;    <SETTYPE abbrevid=4 op0=21/>
;    <NULL/>
;    <SETTYPE abbrevid=4 op0=1/>
;    <CE_CAST abbrevid=6 op0=11 op1=17 op2=16/>
;    <CE_INBOUNDS_GEP op0=6 op1=7 op2=21 op3=18 op4=21 op5=18/>
;    <CE_CAST abbrevid=6 op0=11 op1=15 op2=15/>
;    <CE_CAST abbrevid=6 op0=11 op1=13 op2=8/>
;    <CE_CAST abbrevid=6 op0=11 op1=2 op2=9/>
;    <SETTYPE abbrevid=4 op0=3/>
;    <CSTRING abbrevid=11 op0=112 op1=114 op2=105 op3=110 op4=116/>
;    <SETTYPE abbrevid=4 op0=5/>
;    <CSTRING abbrevid=11 op0=115 op1=97 op2=121 op3=72 op4=105 op5=87 op6=105 op7=116 op8=104/>
;    <SETTYPE abbrevid=4 op0=7/>
;    <CSTRING abbrevid=11 op0=110 op1=101 op2=119/>
;    <SETTYPE abbrevid=4 op0=9/>
;    <CSTRING abbrevid=11 op0=109 op1=97 op2=105 op3=110/>
;    <SETTYPE abbrevid=4 op0=11/>
;    <AGGREGATE abbrevid=8 op0=31 op1=19/>
;    <AGGREGATE abbrevid=8 op0=20 op1=21/>
;    <SETTYPE abbrevid=4 op0=12/>
;    <AGGREGATE abbrevid=8 op0=28 op1=29/>
;    <SETTYPE abbrevid=4 op0=1/>
;    <CE_INBOUNDS_GEP op0=10 op1=6 op2=21 op3=18 op4=21 op5=18/>
;  </CONSTANTS_BLOCK>
;  <METADATA_BLOCK NumWords=23 BlockCodeSize=3>
;    <METADATA_KIND op0=0 op1=100 op2=98 op3=103/>
;    <METADATA_KIND op0=1 op1=116 op2=98 op3=97 op4=97/>
;    <METADATA_KIND op0=2 op1=112 op2=114 op3=111 op4=102/>
;    <METADATA_KIND op0=3 op1=102 op2=112 op3=109 op4=97 op5=116 op6=104/>
;    <METADATA_KIND op0=4 op1=114 op2=97 op3=110 op4=103 op5=101/>
;    <METADATA_KIND op0=5 op1=116 op2=98 op3=97 op4=97 op5=46 op6=115 op7=116 op8=114 op9=117 op10=99 op11=116/>
;    <METADATA_KIND op0=6 op1=105 op2=110 op3=118 op4=97 op5=114 op6=105 op7=97 op8=110 op9=116 op10=46 op11=108 op12=111 op13=97 op14=100/>
;  </METADATA_BLOCK>
;  <VALUE_SYMTAB NumWords=29 BlockCodeSize=4>
;    <ENTRY abbrevid=6 op0=16 op1=101 op2=120 op3=97 op4=109 op5=112 op6=108 op7=101 op8=95 op9=109 op10=97 op11=105 op12=110/>
;    <ENTRY abbrevid=6 op0=1 op1=99 op2=111 op3=110 op4=115 op5=111 op6=108 op7=101/>
;    <ENTRY abbrevid=6 op0=2 op1=103 op2=114 op3=101 op4=101 op5=116 op6=105 op7=110 op8=103/>
;    <ENTRY abbrevid=6 op0=15 op1=101 op2=120 op3=97 op4=109 op5=112 op6=108 op7=101 op8=95 op9=115 op10=97 op11=121 op12=72 op13=105 op14=87 op15=105 op16=116 op17=104/>
;    <ENTRY abbrevid=6 op0=0 op1=115 op2=116 op3=114 op4=105 op5=110 op6=103/>
;    <ENTRY abbrevid=6 op0=14 op1=109 op2=97 op3=108 op4=108 op5=111 op6=99/>
;    <ENTRY abbrevid=6 op0=8 op1=101 op2=120 op3=97 op4=109 op5=112 op6=108 op7=101 op8=95 op9=118 op10=116 op11=97 op12=98/>
;    <ENTRY abbrevid=6 op0=13 op1=115 op2=116 op3=114 op4=105 op5=110 op6=103 op7=95 op8=115 op9=116 op10=114 op11=105 op12=110 op13=103 op14=76 op15=105 op16=116 op17=101 op18=114 op19=97 op20=108/>
;    <ENTRY abbrevid=6 op0=9 op1=95 op2=95 op3=101 op4=120 op5=97 op6=109 op7=112 op8=108 op9=101/>
;    <ENTRY abbrevid=6 op0=12 op1=103 op2=101 op3=116 op4=102 op5=117 op6=110 op7=99/>
;    <ENTRY abbrevid=6 op0=10 op1=101 op2=120 op3=97 op4=109 op5=112 op6=108 op7=101/>
;    <ENTRY abbrevid=6 op0=17 op1=109 op2=97 op3=105 op4=110/>
;  </VALUE_SYMTAB>
;  <FUNCTION_BLOCK NumWords=18 BlockCodeSize=4>
;    <DECLAREBLOCKS op0=1/>
;    <CONSTANTS_BLOCK NumWords=3 BlockCodeSize=4>
;      <SETTYPE abbrevid=4 op0=1/>
;      <CE_INBOUNDS_GEP op0=4 op1=3 op2=21 op3=18 op4=21 op5=18/>
;    </CONSTANTS_BLOCK>
;    <INST_LOAD abbrevid=4 op0=34 op1=0 op2=0/>
;    <INST_CALL op0=0 op1=0 op2=24 op3=1 op4=2/>
;    <INST_CAST abbrevid=7 op0=1 op1=15 op2=11/>
;    <INST_CALL op0=0 op1=0 op2=1 op3=3 op4=5/>
;    <INST_RET abbrevid=9 op0=1/>
;    <VALUE_SYMTAB NumWords=4 BlockCodeSize=4>
;      <BBENTRY abbrevid=7 op0=0 op1=101 op2=110 op3=116 op4=114 op5=121/>
;      <ENTRY abbrevid=6 op0=33 op1=115 op2=97 op3=121 op4=105 op5=110 op6=103/>
;    </VALUE_SYMTAB>
;  </FUNCTION_BLOCK>
;  <FUNCTION_BLOCK NumWords=23 BlockCodeSize=4>
;    <DECLAREBLOCKS op0=1/>
;    <CONSTANTS_BLOCK NumWords=4 BlockCodeSize=4>
;      <SETTYPE abbrevid=4 op0=1/>
;      <CE_INBOUNDS_GEP op0=6 op1=4 op2=21 op3=18 op4=21 op5=18/>
;      <CE_INBOUNDS_GEP op0=8 op1=5 op2=21 op3=18 op4=21 op5=18/>
;    </CONSTANTS_BLOCK>
;    <INST_LOAD op0=4294966291 op1=2 op2=0 op3=0/>
;    <INST_CALL op0=0 op1=0 op2=24 op3=1 op4=3/>
;    <INST_CAST abbrevid=7 op0=1 op1=15 op2=11/>
;    <INST_LOAD abbrevid=4 op0=36 op1=0 op2=0/>
;    <INST_CALL op0=0 op1=0 op2=27 op3=1 op4=5/>
;    <INST_CAST abbrevid=7 op0=1 op1=17 op2=11/>
;    <INST_CALL op0=0 op1=0 op2=1 op3=3/>
;    <INST_CALL op0=0 op1=0 op2=5 op3=7 op4=1/>
;    <INST_RET abbrevid=9 op0=1/>
;    <VALUE_SYMTAB NumWords=2 BlockCodeSize=4>
;      <BBENTRY abbrevid=7 op0=0 op1=101 op2=110 op3=116 op4=114 op5=121/>
;    </VALUE_SYMTAB>
;  </FUNCTION_BLOCK>
;  <FUNCTION_BLOCK NumWords=15 BlockCodeSize=4>
;    <DECLAREBLOCKS op0=1/>
;    <CONSTANTS_BLOCK NumWords=3 BlockCodeSize=4>
;      <SETTYPE abbrevid=4 op0=1/>
;      <CE_INBOUNDS_GEP op0=10 op1=11 op2=21 op3=18 op4=21 op5=18/>
;    </CONSTANTS_BLOCK>
;    <INST_LOAD abbrevid=4 op0=23 op1=0 op2=0/>
;    <INST_CALL op0=0 op1=0 op2=22 op3=1 op4=2/>
;    <INST_CAST abbrevid=7 op0=1 op1=17 op2=11/>
;    <INST_CALL op0=0 op1=0 op2=1 op3=3/>
;    <INST_RET abbrevid=9 op0=19/>
;    <VALUE_SYMTAB NumWords=2 BlockCodeSize=4>
;      <BBENTRY abbrevid=7 op0=0 op1=101 op2=110 op3=116 op4=114 op5=121/>
;    </VALUE_SYMTAB>
;  </FUNCTION_BLOCK>
;</MODULE_BLOCK>
