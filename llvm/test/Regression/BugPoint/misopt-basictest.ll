; RUN: PATH=/usr/bin:/bin/:${PATH}
; RUN: export PATH
; RUN: bugpoint %s -dce -bugpoint-deletecalls -simplifycfg

%.LC0 = internal global [13 x sbyte] c"Hello World\0A\00"

implementation

declare int %printf(sbyte*, ...)

int %main() {
        call int(sbyte*, ...)* %printf( sbyte* getelementptr ([13 x sbyte]* %.LC0, long 0, long 0) )
        ret int 0
}

