RUN: echo TA > %T/TA.txt
RUN: echo TB > %T/TB.txt
RUN: echo TAB > %T/TAB.txt

RUN: echo %T/TA* | FileCheck -check-prefix=STAR %s
RUN: echo %T/'TA'* | FileCheck -check-prefix=STAR %s
RUN: echo %T/T'A'* | FileCheck -check-prefix=STAR %s

RUN: echo %T/T?.txt | FileCheck -check-prefix=QUESTION %s
RUN: echo %T/'T'?.txt | FileCheck -check-prefix=QUESTION %s

RUN: echo %T/T??.txt | FileCheck -check-prefix=QUESTION2 %s
RUN: echo %T/'T'??.txt | FileCheck -check-prefix=QUESTION2 %s

RUN: echo 'T*' 'T?.txt' 'T??.txt' | FileCheck -check-prefix=QUOTEDARGS %s

STAR-NOT: TB.txt
STAR: {{(TA.txt.*TAB.txt|TAB.txt.*TA.txt)}}

QUESTION-NOT: TAB.txt
QUESTION: {{(TA.txt.*TB.txt|TB.txt.*TA.txt)}}

QUESTION2-NOT: TA.txt
QUESTION2-NOT: TB.txt
QUESTION2: TAB.txt

QUOTEDARGS-NOT: .txt
QUOTEDARGS: T* T?.txt T??.txt
