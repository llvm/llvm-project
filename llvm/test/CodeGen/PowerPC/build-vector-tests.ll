; RUN: llc -mcpu=pwr9 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   -mtriple=powerpc64-unknown-unknown < %s | FileCheck %s \
; RUN:   -check-prefix=P9BE -implicit-check-not frsp
; RUN: llc -mcpu=pwr9 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s \
; RUN:   -check-prefix=P9LE -implicit-check-not frsp
; RUN: llc -mcpu=pwr8 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   -mtriple=powerpc64-unknown-unknown < %s | FileCheck %s \
; RUN:   -check-prefix=P8BE -implicit-check-not frsp
; RUN: llc -mcpu=pwr8 -ppc-asm-full-reg-names -ppc-vsr-nums-as-vr \
; RUN:   -mtriple=powerpc64le-unknown-unknown < %s | FileCheck %s \
; RUN:   -check-prefix=P8LE -implicit-check-not frsp

; This test case comes from the following C test case (included as it may be
; slightly more readable than the LLVM IR.

;/*  This test case provides various ways of building vectors to ensure we
;    produce optimal code for all cases. The cases are (for each type):
;    - All zeros
;    - All ones
;    - Splat of a constant
;    - From different values already in registers
;    - From different constants
;    - From different values in memory
;    - Splat of a value in register
;    - Splat of a value in memory
;    - Inserting element into existing vector
;    - Inserting element from existing vector into existing vector
;
;    With conversions (float <-> int)
;    - Splat of a constant
;    - From different values already in registers
;    - From different constants
;    - From different values in memory
;    - Splat of a value in register
;    - Splat of a value in memory
;    - Inserting element into existing vector
;    - Inserting element from existing vector into existing vector
;*/
;
;/*=================================== int ===================================*/
;// P8: xxlxor                                                                //
;// P9: xxlxor                                                                //
;vector int allZeroi() {                                                      //
;  return (vector int)0;                                                      //
;}                                                                            //
;// P8: vspltisb -1                                                           //
;// P9: xxspltisb 255                                                         //
;vector int allOnei() {                                                       //
;  return (vector int)-1;                                                     //
;}                                                                            //
;// P8: vspltisw 1                                                            //
;// P9: vspltisw 1                                                            //
;vector int spltConst1i() {                                                   //
;  return (vector int)1;                                                      //
;}                                                                            //
;// P8: vspltisw -15; vsrw                                                    //
;// P9: vspltisw -15; vsrw                                                    //
;vector int spltConst16ki() {                                                 //
;  return (vector int)((1<<15) - 1);                                          //
;}                                                                            //
;// P8: vspltisw -16; vsrw                                                    //
;// P9: vspltisw -16; vsrw                                                    //
;vector int spltConst32ki() {                                                 //
;  return (vector int)((1<<16) - 1);                                          //
;}                                                                            //
;// P8: 4 x mtvsrwz, 2 x xxmrgh, vmrgow                                       //
;// P9: 2 x mtvsrdd, vmrgow                                                   //
;vector int fromRegsi(int a, int b, int c, int d) {                           //
;  return (vector int){ a, b, c, d };                                         //
;}                                                                            //
;// P8: lxvd2x, xxswapd                                                       //
;// P9: lxvx (or even lxv)                                                    //
;vector int fromDiffConstsi() {                                               //
;  return (vector int) { 242, -113, 889, 19 };                                //
;}                                                                            //
;// P8: lxvd2x, xxswapd                                                       //
;// P9: lxvx                                                                  //
;vector int fromDiffMemConsAi(int *arr) {                                     //
;  return (vector int) { arr[0], arr[1], arr[2], arr[3] };                    //
;}                                                                            //
;// P8: 2 x lxvd2x, 2 x xxswapd, vperm                                        //
;// P9: 2 x lxvx, vperm                                                       //
;vector int fromDiffMemConsDi(int *arr) {                                     //
;  return (vector int) { arr[3], arr[2], arr[1], arr[0] };                    //
;}                                                                            //
;// P8: sldi 2, lxvd2x, xxswapd                                               //
;// P9: sldi 2, lxvx                                                          //
;vector int fromDiffMemVarAi(int *arr, int elem) {                            //
;  return (vector int) { arr[elem], arr[elem+1], arr[elem+2], arr[elem+3] };  //
;}                                                                            //
;// P8: sldi 2, 2 x lxvd2x, 2 x xxswapd, vperm                                //
;// P9: sldi 2, 2 x lxvx, vperm                                               //
;vector int fromDiffMemVarDi(int *arr, int elem) {                            //
;  return (vector int) { arr[elem], arr[elem-1], arr[elem-2], arr[elem-3] };  //
;}                                                                            //
;// P8: 4 x lwz, 4 x mtvsrwz, 2 x xxmrghd, vmrgow                             //
;// P9: 4 x lwz, 2 x mtvsrdd, vmrgow                                          //
;vector int fromRandMemConsi(int *arr) {                                      //
;  return (vector int) { arr[4], arr[18], arr[2], arr[88] };                  //
;}                                                                            //
;// P8: sldi 2, 4 x lwz, 4 x mtvsrwz, 2 x xxmrghd, vmrgow                     //
;// P9: sldi 2, add, 4 x lwz, 2 x mtvsrdd, vmrgow                             //
;vector int fromRandMemVari(int *arr, int elem) {                             //
;  return (vector int) { arr[elem+4], arr[elem+1], arr[elem+2], arr[elem+8] };//
;}                                                                            //
;// P8: mtvsrwz, xxspltw                                                      //
;// P9: mtvsrws                                                               //
;vector int spltRegVali(int val) {                                            //
;  return (vector int) val;                                                   //
;}                                                                            //
;// P8: lxsiwax, xxspltw                                                      //
;// P9: lxvwsx                                                                //
;vector int spltMemVali(int *ptr) {                                           //
;  return (vector int)*ptr;                                                   //
;}                                                                            //
;// P8: vspltisw                                                              //
;// P9: vspltisw                                                              //
;vector int spltCnstConvftoi() {                                              //
;  return (vector int) 4.74f;                                                 //
;}                                                                            //
;// P8: 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspsxws                         //
;// P9: 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvdpsxws                         //
;vector int fromRegsConvftoi(float a, float b, float c, float d) {            //
;  return (vector int) { a, b, c, d };                                        //
;}                                                                            //
;// P8: lxvd2x, xxswapd                                                       //
;// P9: lxvx (even lxv)                                                       //
;vector int fromDiffConstsConvftoi() {                                        //
;  return (vector int) { 24.46f, 234.f, 988.19f, 422.39f };                   //
;}                                                                            //
;// P8: lxvd2x, xxswapd, xvcvspsxws                                           //
;// P9: lxvx, xvcvspsxws                                                      //
;vector int fromDiffMemConsAConvftoi(float *ptr) {                            //
;  return (vector int) { ptr[0], ptr[1], ptr[2], ptr[3] };                    //
;}                                                                            //
;// P8: 2 x lxvd2x, 2 x xxswapd, vperm, xvcvspsxws                            //
;// P9: 2 x lxvx, vperm, xvcvspsxws                                           //
;vector int fromDiffMemConsDConvftoi(float *ptr) {                            //
;  return (vector int) { ptr[3], ptr[2], ptr[1], ptr[0] };                    //
;}                                                                            //
;// P8: 4 x lxsspx, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspsxws             //
;// P9: 4 x lxssp, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspsxws              //
;// Note: if the consecutive loads learns to handle pre-inc, this can be:     //
;//       sldi 2, load, xvcvspuxws                                            //
;vector int fromDiffMemVarAConvftoi(float *arr, int elem) {                   //
;  return (vector int) { arr[elem], arr[elem+1], arr[elem+2], arr[elem+3] };  //
;}                                                                            //
;// P8: 4 x lxsspx, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspsxws             //
;// P9: 4 x lxssp, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspsxws              //
;// Note: if the consecutive loads learns to handle pre-inc, this can be:     //
;//       sldi 2, 2 x load, vperm, xvcvspuxws                                 //
;vector int fromDiffMemVarDConvftoi(float *arr, int elem) {                   //
;  return (vector int) { arr[elem], arr[elem-1], arr[elem-2], arr[elem-3] };  //
;}                                                                            //
;// P8: xscvdpsxws, xxspltw                                                   //
;// P9: xscvdpsxws, xxspltw                                                   //
;vector int spltRegValConvftoi(float val) {                                   //
;  return (vector int) val;                                                   //
;}                                                                            //
;// P8: lxsspx, xscvdpsxws, xxspltw                                           //
;// P9: lxvwsx, xvcvspsxws                                                    //
;vector int spltMemValConvftoi(float *ptr) {                                  //
;  return (vector int)*ptr;                                                   //
;}                                                                            //
;// P8: vspltisw                                                              //
;// P9: vspltisw                                                              //
;vector int spltCnstConvdtoi() {                                              //
;  return (vector int) 4.74;                                                  //
;}                                                                            //
;// P8: 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspsxws                         //
;// P9: 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspsxws                         //
;vector int fromRegsConvdtoi(double a, double b, double c, double d) {        //
;  return (vector int) { a, b, c, d };                                        //
;}                                                                            //
;// P8: lxvd2x, xxswapd                                                       //
;// P9: lxvx (even lxv)                                                       //
;vector int fromDiffConstsConvdtoi() {                                        //
;  return (vector int) { 24.46, 234., 988.19, 422.39 };                       //
;}                                                                            //
;// P8: 2 x lxvd2x, 2 x xxswapd, xxmrgld, xxmrghd, 2 x xvcvdpsp, vmrgew,      //
;//     xvcvspsxws                                                            //
;// P9: 2 x lxvx, 2 x xxswapd, xxmrgld, xxmrghd, 2 x xvcvdpsp, vmrgew,        //
;//     xvcvspsxws                                                            //
;vector int fromDiffMemConsAConvdtoi(double *ptr) {                           //
;  return (vector int) { ptr[0], ptr[1], ptr[2], ptr[3] };                    //
;}                                                                            //
;// P8: 4 x lxsdx, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspsxws              //
;// P9: 4 x lfd, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspsxws                //
;vector int fromDiffMemConsDConvdtoi(double *ptr) {                           //
;  return (vector int) { ptr[3], ptr[2], ptr[1], ptr[0] };                    //
;}                                                                            //
;// P8: lfdux, 3 x lxsdx, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspsxws       //
;// P9: lfdux, 3 x lfd, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspsxws         //
;vector int fromDiffMemVarAConvdtoi(double *arr, int elem) {                  //
;  return (vector int) { arr[elem], arr[elem+1], arr[elem+2], arr[elem+3] };  //
;}                                                                            //
;// P8: lfdux, 3 x lxsdx, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspsxws       //
;// P9: lfdux, 3 x lfd, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspsxws         //
;vector int fromDiffMemVarDConvdtoi(double *arr, int elem) {                  //
;  return (vector int) { arr[elem], arr[elem-1], arr[elem-2], arr[elem-3] };  //
;}                                                                            //
;// P8: xscvdpsxws, xxspltw                                                   //
;// P9: xscvdpsxws, xxspltw                                                   //
;vector int spltRegValConvdtoi(double val) {                                  //
;  return (vector int) val;                                                   //
;}                                                                            //
;// P8: lxsdx, xscvdpsxws, xxspltw                                            //
;// P9: lxssp, xscvdpsxws, xxspltw                                            //
;vector int spltMemValConvdtoi(double *ptr) {                                 //
;  return (vector int)*ptr;                                                   //
;}                                                                            //
;/*=================================== int ===================================*/
;/*=============================== unsigned int ==============================*/
;// P8: xxlxor                                                                //
;// P9: xxlxor                                                                //
;vector unsigned int allZeroui() {                                            //
;  return (vector unsigned int)0;                                             //
;}                                                                            //
;// P8: vspltisb -1                                                           //
;// P9: xxspltisb 255                                                         //
;vector unsigned int allOneui() {                                             //
;  return (vector unsigned int)-1;                                            //
;}                                                                            //
;// P8: vspltisw 1                                                            //
;// P9: vspltisw 1                                                            //
;vector unsigned int spltConst1ui() {                                         //
;  return (vector unsigned int)1;                                             //
;}                                                                            //
;// P8: vspltisw -15; vsrw                                                    //
;// P9: vspltisw -15; vsrw                                                    //
;vector unsigned int spltConst16kui() {                                       //
;  return (vector unsigned int)((1<<15) - 1);                                 //
;}                                                                            //
;// P8: vspltisw -16; vsrw                                                    //
;// P9: vspltisw -16; vsrw                                                    //
;vector unsigned int spltConst32kui() {                                       //
;  return (vector unsigned int)((1<<16) - 1);                                 //
;}                                                                            //
;// P8: 4 x mtvsrwz, 2 x xxmrghd, vmrgow                                      //
;// P9: 2 x mtvsrdd, vmrgow                                                   //
;vector unsigned int fromRegsui(unsigned int a, unsigned int b,               //
;                              unsigned int c, unsigned int d) {              //
;  return (vector unsigned int){ a, b, c, d };                                //
;}                                                                            //
;// P8: lxvd2x, xxswapd                                                       //
;// P9: lxvx (or even lxv)                                                    //
;vector unsigned int fromDiffConstsui() {                                     //
;  return (vector unsigned int) { 242, -113, 889, 19 };                       //
;}                                                                            //
;// P8: lxvd2x, xxswapd                                                       //
;// P9: lxvx                                                                  //
;vector unsigned int fromDiffMemConsAui(unsigned int *arr) {                  //
;  return (vector unsigned int) { arr[0], arr[1], arr[2], arr[3] };           //
;}                                                                            //
;// P8: 2 x lxvd2x, 2 x xxswapd, vperm                                        //
;// P9: 2 x lxvx, vperm                                                       //
;vector unsigned int fromDiffMemConsDui(unsigned int *arr) {                  //
;  return (vector unsigned int) { arr[3], arr[2], arr[1], arr[0] };           //
;}                                                                            //
;// P8: sldi 2, lxvd2x, xxswapd                                               //
;// P9: sldi 2, lxvx                                                          //
;vector unsigned int fromDiffMemVarAui(unsigned int *arr, int elem) {         //
;  return (vector unsigned int) { arr[elem], arr[elem+1],                     //
;                                 arr[elem+2], arr[elem+3] };                 //
;}                                                                            //
;// P8: sldi 2, 2 x lxvd2x, 2 x xxswapd, vperm                                //
;// P9: sldi 2, 2 x lxvx, vperm                                               //
;vector unsigned int fromDiffMemVarDui(unsigned int *arr, int elem) {         //
;  return (vector unsigned int) { arr[elem], arr[elem-1],                     //
;                                 arr[elem-2], arr[elem-3] };                 //
;}                                                                            //
;// P8: 4 x lwz, 4 x mtvsrwz, 2 x xxmrghd, vmrgow                             //
;// P9: 4 x lwz, 2 x mtvsrdd, vmrgow                                          //
;vector unsigned int fromRandMemConsui(unsigned int *arr) {                   //
;  return (vector unsigned int) { arr[4], arr[18], arr[2], arr[88] };         //
;}                                                                            //
;// P8: sldi 2, 4 x lwz, 4 x mtvsrwz, 2 x xxmrghd, vmrgow                     //
;// P9: sldi 2, add, 4 x lwz, 2 x mtvsrdd, vmrgow                             //
;vector unsigned int fromRandMemVarui(unsigned int *arr, int elem) {          //
;  return (vector unsigned int) { arr[elem+4], arr[elem+1],                   //
;                                 arr[elem+2], arr[elem+8] };                 //
;}                                                                            //
;// P8: mtvsrwz, xxspltw                                                      //
;// P9: mtvsrws                                                               //
;vector unsigned int spltRegValui(unsigned int val) {                         //
;  return (vector unsigned int) val;                                          //
;}                                                                            //
;// P8: lxsiwax, xxspltw                                                      //
;// P9: lxvwsx                                                                //
;vector unsigned int spltMemValui(unsigned int *ptr) {                        //
;  return (vector unsigned int)*ptr;                                          //
;}                                                                            //
;// P8: vspltisw                                                              //
;// P9: vspltisw                                                              //
;vector unsigned int spltCnstConvftoui() {                                    //
;  return (vector unsigned int) 4.74f;                                        //
;}                                                                            //
;// P8: 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspuxws                         //
;// P9: 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspuxws                         //
;vector unsigned int fromRegsConvftoui(float a, float b, float c, float d) {  //
;  return (vector unsigned int) { a, b, c, d };                               //
;}                                                                            //
;// P8: lxvd2x, xxswapd                                                       //
;// P9: lxvx (even lxv)                                                       //
;vector unsigned int fromDiffConstsConvftoui() {                              //
;  return (vector unsigned int) { 24.46f, 234.f, 988.19f, 422.39f };          //
;}                                                                            //
;// P8: lxvd2x, xxswapd, xvcvspuxws                                           //
;// P9: lxvx, xvcvspuxws                                                      //
;vector unsigned int fromDiffMemConsAConvftoui(float *ptr) {                  //
;  return (vector unsigned int) { ptr[0], ptr[1], ptr[2], ptr[3] };           //
;}                                                                            //
;// P8: 2 x lxvd2x, 2 x xxswapd, vperm, xvcvspuxws                            //
;// P9: 2 x lxvx, vperm, xvcvspuxws                                           //
;vector unsigned int fromDiffMemConsDConvftoui(float *ptr) {                  //
;  return (vector unsigned int) { ptr[3], ptr[2], ptr[1], ptr[0] };           //
;}                                                                            //
;// P8: lfsux, 3 x lxsspx, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspuxws      //
;// P9: lfsux, 3 x lfs, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspuxws         //
;// Note: if the consecutive loads learns to handle pre-inc, this can be:     //
;//       sldi 2, load, xvcvspuxws                                            //
;vector unsigned int fromDiffMemVarAConvftoui(float *arr, int elem) {         //
;  return (vector unsigned int) { arr[elem], arr[elem+1],                     //
;                                 arr[elem+2], arr[elem+3] };                 //
;}                                                                            //
;// P8: lfsux, 3 x lxsspx, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspuxws      //
;// P9: lfsux, 3 x lfs, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspuxws         //
;// Note: if the consecutive loads learns to handle pre-inc, this can be:     //
;//       sldi 2, 2 x load, vperm, xvcvspuxws                                 //
;vector unsigned int fromDiffMemVarDConvftoui(float *arr, int elem) {         //
;  return (vector unsigned int) { arr[elem], arr[elem-1],                     //
;                                 arr[elem-2], arr[elem-3] };                 //
;}                                                                            //
;// P8: xscvdpuxws, xxspltw                                                   //
;// P9: xscvdpuxws, xxspltw                                                   //
;vector unsigned int spltRegValConvftoui(float val) {                         //
;  return (vector unsigned int) val;                                          //
;}                                                                            //
;// P8: lxsspx, xscvdpuxws, xxspltw                                           //
;// P9: lxvwsx, xvcvspuxws                                                    //
;vector unsigned int spltMemValConvftoui(float *ptr) {                        //
;  return (vector unsigned int)*ptr;                                          //
;}                                                                            //
;// P8: vspltisw                                                              //
;// P9: vspltisw                                                              //
;vector unsigned int spltCnstConvdtoui() {                                    //
;  return (vector unsigned int) 4.74;                                         //
;}                                                                            //
;// P8: 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspuxws                         //
;// P9: 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspuxws                         //
;vector unsigned int fromRegsConvdtoui(double a, double b,                    //
;                                      double c, double d) {                  //
;  return (vector unsigned int) { a, b, c, d };                               //
;}                                                                            //
;// P8: lxvd2x, xxswapd                                                       //
;// P9: lxvx (even lxv)                                                       //
;vector unsigned int fromDiffConstsConvdtoui() {                              //
;  return (vector unsigned int) { 24.46, 234., 988.19, 422.39 };              //
;}                                                                            //
;// P8: 2 x lxvd2x, 2 x xxswapd, xxmrgld, xxmrghd, 2 x xvcvdpsp, vmrgew,      //
;//     xvcvspuxws                                                            //
;// P9: 2 x lxvx, xxmrgld, xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspuxws          //
;vector unsigned int fromDiffMemConsAConvdtoui(double *ptr) {                 //
;  return (vector unsigned int) { ptr[0], ptr[1], ptr[2], ptr[3] };           //
;}                                                                            //
;// P8: 4 x lxsdx, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspuxws              //
;// P9: 4 x lfd, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspuxws                //
;vector unsigned int fromDiffMemConsDConvdtoui(double *ptr) {                 //
;  return (vector unsigned int) { ptr[3], ptr[2], ptr[1], ptr[0] };           //
;}                                                                            //
;// P8: lfdux, 3 x lxsdx, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspuxws       //
;// P9: lfdux, 3 x lfd, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspuxws         //
;vector unsigned int fromDiffMemVarAConvdtoui(double *arr, int elem) {        //
;  return (vector unsigned int) { arr[elem], arr[elem+1],                     //
;                                 arr[elem+2], arr[elem+3] };                 //
;}                                                                            //
;// P8: lfdux, 3 x lxsdx, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspuxws       //
;// P9: lfdux, 3 x lfd, 2 x xxmrghd, 2 x xvcvdpsp, vmrgew, xvcvspuxws         //
;vector unsigned int fromDiffMemVarDConvdtoui(double *arr, int elem) {        //
;  return (vector unsigned int) { arr[elem], arr[elem-1],                     //
;                                 arr[elem-2], arr[elem-3] };                 //
;}                                                                            //
;// P8: xscvdpuxws, xxspltw                                                   //
;// P9: xscvdpuxws, xxspltw                                                   //
;vector unsigned int spltRegValConvdtoui(double val) {                        //
;  return (vector unsigned int) val;                                          //
;}                                                                            //
;// P8: lxsspx, xscvdpuxws, xxspltw                                           //
;// P9: lfd, xscvdpuxws, xxspltw                                              //
;vector unsigned int spltMemValConvdtoui(double *ptr) {                       //
;  return (vector unsigned int)*ptr;                                          //
;}                                                                            //
;/*=============================== unsigned int ==============================*/
;/*=============================== long long =================================*/
;// P8: xxlxor                                                                //
;// P9: xxlxor                                                                //
;vector long long allZeroll() {                                               //
;  return (vector long long)0;                                                //
;}                                                                            //
;// P8: vspltisb -1                                                           //
;// P9: xxspltisb 255                                                         //
;vector long long allOnell() {                                                //
;  return (vector long long)-1;                                               //
;}                                                                            //
;// P8: constant pool load (possible: vmrgew (xxlxor), (vspltisw))            //
;// P9: constant pool load (possible: vmrgew (xxlxor), (vspltisw))            //
;vector long long spltConst1ll() {                                            //
;  return (vector long long)1;                                                //
;}                                                                            //
;// P8: constant pool load (possible: vmrgew (xxlxor), (vspltisw, vsrw))      //
;// P9: constant pool load (possible: vmrgew (xxlxor), (vspltisw, vsrw))      //
;vector long long spltConst16kll() {                                          //
;  return (vector long long)((1<<15) - 1);                                    //
;}                                                                            //
;// P8: constant pool load (possible: vmrgew (xxlxor), (vspltisw, vsrw))      //
;// P9: constant pool load (possible: vmrgew (xxlxor), (vspltisw, vsrw))      //
;vector long long spltConst32kll() {                                          //
;  return (vector long long)((1<<16) - 1);                                    //
;}                                                                            //
;// P8: 2 x mtvsrd, xxmrghd                                                   //
;// P9: mtvsrdd                                                               //
;vector long long fromRegsll(long long a, long long b) {                      //
;  return (vector long long){ a, b };                                         //
;}                                                                            //
;// P8: lxvd2x, xxswapd                                                       //
;// P9: lxvx (or even lxv)                                                    //
;vector long long fromDiffConstsll() {                                        //
;  return (vector long long) { 242, -113 };                                   //
;}                                                                            //
;// P8: lxvd2x, xxswapd                                                       //
;// P9: lxvx                                                                  //
;vector long long fromDiffMemConsAll(long long *arr) {                        //
;  return (vector long long) { arr[0], arr[1] };                              //
;}                                                                            //
;// P8: lxvd2x                                                                //
;// P9: lxvx, xxswapd (maybe just use lxvd2x)                                 //
;vector long long fromDiffMemConsDll(long long *arr) {                        //
;  return (vector long long) { arr[3], arr[2] };                              //
;}                                                                            //
;// P8: sldi 3, lxvd2x, xxswapd                                               //
;// P9: sldi 3, lxvx                                                          //
;vector long long fromDiffMemVarAll(long long *arr, int elem) {               //
;  return (vector long long) { arr[elem], arr[elem+1] };                      //
;}                                                                            //
;// P8: sldi 3, lxvd2x                                                        //
;// P9: sldi 3, lxvx, xxswapd (maybe just use lxvd2x)                         //
;vector long long fromDiffMemVarDll(long long *arr, int elem) {               //
;  return (vector long long) { arr[elem], arr[elem-1] };                      //
;}                                                                            //
;// P8: 2 x ld, 2 x mtvsrd, xxmrghd                                           //
;// P9: 2 x ld, mtvsrdd                                                       //
;vector long long fromRandMemConsll(long long *arr) {                         //
;  return (vector long long) { arr[4], arr[18] };                             //
;}                                                                            //
;// P8: sldi 3, add, 2 x ld, 2 x mtvsrd, xxmrghd                              //
;// P9: sldi 3, add, 2 x ld, mtvsrdd                                          //
;vector long long fromRandMemVarll(long long *arr, int elem) {                //
;  return (vector long long) { arr[elem+4], arr[elem+1] };                    //
;}                                                                            //
;// P8: mtvsrd, xxspltd                                                       //
;// P9: mtvsrdd                                                               //
;vector long long spltRegValll(long long val) {                               //
;  return (vector long long) val;                                             //
;}                                                                            //
;// P8: lxvdsx                                                                //
;// P9: lxvdsx                                                                //
;vector long long spltMemValll(long long *ptr) {                              //
;  return (vector long long)*ptr;                                             //
;}                                                                            //
;// P8: constant pool load (possible: vmrgew (xxlxor), (vspltisw))            //
;// P9: constant pool load (possible: vmrgew (xxlxor), (vspltisw))            //
;vector long long spltCnstConvftoll() {                                       //
;  return (vector long long) 4.74f;                                           //
;}                                                                            //
;// P8: xxmrghd, xvcvdpsxds                                                   //
;// P9: xxmrghd, xvcvdpsxds                                                   //
;vector long long fromRegsConvftoll(float a, float b) {                       //
;  return (vector long long) { a, b };                                        //
;}                                                                            //
;// P8: lxvd2x, xxswapd                                                       //
;// P9: lxvx (even lxv)                                                       //
;vector long long fromDiffConstsConvftoll() {                                 //
;  return (vector long long) { 24.46f, 234.f };                               //
;}                                                                            //
;// P8: 2 x lxsspx, xxmrghd, xvcvdpsxds                                       //
;// P9: 2 x lxssp, xxmrghd, xvcvdpsxds                                        //
;vector long long fromDiffMemConsAConvftoll(float *ptr) {                     //
;  return (vector long long) { ptr[0], ptr[1] };                              //
;}                                                                            //
;// P8: 2 x lxsspx, xxmrghd, xvcvdpsxds                                       //
;// P9: 2 x lxssp, xxmrghd, xvcvdpsxds                                        //
;vector long long fromDiffMemConsDConvftoll(float *ptr) {                     //
;  return (vector long long) { ptr[3], ptr[2] };                              //
;}                                                                            //
;// P8: sldi 2, lfsux, lxsspx, xxmrghd, xvcvdpsxds                            //
;// P9: sldi 2, lfsux, lfs, xxmrghd, xvcvdpsxds                               //
;vector long long fromDiffMemVarAConvftoll(float *arr, int elem) {            //
;  return (vector long long) { arr[elem], arr[elem+1] };                      //
;}                                                                            //
;// P8: sldi 2, lfsux, lxsspx, xxmrghd, xvcvdpsxds                            //
;// P9: sldi 2, lfsux, lfs, xxmrghd, xvcvdpsxds                               //
;vector long long fromDiffMemVarDConvftoll(float *arr, int elem) {            //
;  return (vector long long) { arr[elem], arr[elem-1] };                      //
;}                                                                            //
;// P8: xscvdpsxds, xxspltd                                                   //
;// P9: xscvdpsxds, xxspltd                                                   //
;vector long long spltRegValConvftoll(float val) {                            //
;  return (vector long long) val;                                             //
;}                                                                            //
;// P8: lxsspx, xscvdpsxds, xxspltd                                           //
;// P9: lfs, xscvdpsxds, xxspltd                                              //
;vector long long spltMemValConvftoll(float *ptr) {                           //
;  return (vector long long)*ptr;                                             //
;}                                                                            //
;// P8: constant pool load (possible: vmrgew (xxlxor), (vspltisw))            //
;// P9: constant pool load (possible: vmrgew (xxlxor), (vspltisw))            //
;vector long long spltCnstConvdtoll() {                                       //
;  return (vector long long) 4.74;                                            //
;}                                                                            //
;// P8: xxmrghd, xvcvdpsxds                                                   //
;// P9: xxmrghd, xvcvdpsxds                                                   //
;vector long long fromRegsConvdtoll(double a, double b) {                     //
;  return (vector long long) { a, b };                                        //
;}                                                                            //
;// P8: lxvd2x, xxswapd                                                       //
;// P9: lxvx (even lxv)                                                       //
;vector long long fromDiffConstsConvdtoll() {                                 //
;  return (vector long long) { 24.46, 234. };                                 //
;}                                                                            //
;// P8: lxvd2x, xxswapd, xvcvdpsxds                                           //
;// P9: lxvx, xvcvdpsxds                                                      //
;vector long long fromDiffMemConsAConvdtoll(double *ptr) {                    //
;  return (vector long long) { ptr[0], ptr[1] };                              //
;}                                                                            //
;// P8: lxvd2x, xvcvdpsxds                                                    //
;// P9: lxvx, xxswapd, xvcvdpsxds                                             //
;vector long long fromDiffMemConsDConvdtoll(double *ptr) {                    //
;  return (vector long long) { ptr[3], ptr[2] };                              //
;}                                                                            //
;// P8: sldi 3, lxvd2x, xxswapd, xvcvdpsxds                                   //
;// P9: sldi 3, lxvx, xvcvdpsxds                                              //
;vector long long fromDiffMemVarAConvdtoll(double *arr, int elem) {           //
;  return (vector long long) { arr[elem], arr[elem+1] };                      //
;}                                                                            //
;// P8: sldi 3, lxvd2x, xvcvdpsxds                                            //
;// P9: sldi 3, lxvx, xxswapd, xvcvdpsxds                                     //
;vector long long fromDiffMemVarDConvdtoll(double *arr, int elem) {           //
;  return (vector long long) { arr[elem], arr[elem-1] };                      //
;}                                                                            //
;// P8: xscvdpsxds, xxspltd                                                   //
;// P9: xscvdpsxds, xxspltd                                                   //
;vector long long spltRegValConvdtoll(double val) {                           //
;  return (vector long long) val;                                             //
;}                                                                            //
;// P8: lxvdsx, xvcvdpsxds                                                    //
;// P9: lxvdsx, xvcvdpsxds                                                    //
;vector long long spltMemValConvdtoll(double *ptr) {                          //
;  return (vector long long)*ptr;                                             //
;}                                                                            //
;/*=============================== long long =================================*/
;/*========================== unsigned long long =============================*/
;// P8: xxlxor                                                                //
;// P9: xxlxor                                                                //
;vector unsigned long long allZeroull() {                                     //
;  return (vector unsigned long long)0;                                       //
;}                                                                            //
;// P8: vspltisb -1                                                           //
;// P9: xxspltisb 255                                                         //
;vector unsigned long long allOneull() {                                      //
;  return (vector unsigned long long)-1;                                      //
;}                                                                            //
;// P8: constant pool load (possible: vmrgew (xxlxor), (vspltisw))            //
;// P9: constant pool load (possible: vmrgew (xxlxor), (vspltisw))            //
;vector unsigned long long spltConst1ull() {                                  //
;  return (vector unsigned long long)1;                                       //
;}                                                                            //
;// P8: constant pool load (possible: vmrgew (xxlxor), (vspltisw, vsrw))      //
;// P9: constant pool load (possible: vmrgew (xxlxor), (vspltisw, vsrw))      //
;vector unsigned long long spltConst16kull() {                                //
;  return (vector unsigned long long)((1<<15) - 1);                           //
;}                                                                            //
;// P8: constant pool load (possible: vmrgew (xxlxor), (vspltisw, vsrw))      //
;// P9: constant pool load (possible: vmrgew (xxlxor), (vspltisw, vsrw))      //
;vector unsigned long long spltConst32kull() {                                //
;  return (vector unsigned long long)((1<<16) - 1);                           //
;}                                                                            //
;// P8: 2 x mtvsrd, xxmrghd                                                   //
;// P9: mtvsrdd                                                               //
;vector unsigned long long fromRegsull(unsigned long long a,                  //
;                                      unsigned long long b) {                //
;  return (vector unsigned long long){ a, b };                                //
;}                                                                            //
;// P8: lxvd2x, xxswapd                                                       //
;// P9: lxvx (or even lxv)                                                    //
;vector unsigned long long fromDiffConstsull() {                              //
;  return (vector unsigned long long) { 242, -113 };                          //
;}                                                                            //
;// P8: lxvd2x, xxswapd                                                       //
;// P9: lxvx                                                                  //
;vector unsigned long long fromDiffMemConsAull(unsigned long long *arr) {     //
;  return (vector unsigned long long) { arr[0], arr[1] };                     //
;}                                                                            //
;// P8: lxvd2x                                                                //
;// P9: lxvx, xxswapd (maybe just use lxvd2x)                                 //
;vector unsigned long long fromDiffMemConsDull(unsigned long long *arr) {     //
;  return (vector unsigned long long) { arr[3], arr[2] };                     //
;}                                                                            //
;// P8: sldi 3, lxvd2x, xxswapd                                               //
;// P9: sldi 3, lxvx                                                          //
;vector unsigned long long fromDiffMemVarAull(unsigned long long *arr,        //
;                                             int elem) {                     //
;  return (vector unsigned long long) { arr[elem], arr[elem+1] };             //
;}                                                                            //
;// P8: sldi 3, lxvd2x                                                        //
;// P9: sldi 3, lxvx, xxswapd (maybe just use lxvd2x)                         //
;vector unsigned long long fromDiffMemVarDull(unsigned long long *arr,        //
;                                             int elem) {                     //
;  return (vector unsigned long long) { arr[elem], arr[elem-1] };             //
;}                                                                            //
;// P8: 2 x ld, 2 x mtvsrd, xxmrghd                                           //
;// P9: 2 x ld, mtvsrdd                                                       //
;vector unsigned long long fromRandMemConsull(unsigned long long *arr) {      //
;  return (vector unsigned long long) { arr[4], arr[18] };                    //
;}                                                                            //
;// P8: sldi 3, add, 2 x ld, 2 x mtvsrd, xxmrghd                              //
;// P9: sldi 3, add, 2 x ld, mtvsrdd                                          //
;vector unsigned long long fromRandMemVarull(unsigned long long *arr,         //
;                                            int elem) {                      //
;  return (vector unsigned long long) { arr[elem+4], arr[elem+1] };           //
;}                                                                            //
;// P8: mtvsrd, xxspltd                                                       //
;// P9: mtvsrdd                                                               //
;vector unsigned long long spltRegValull(unsigned long long val) {            //
;  return (vector unsigned long long) val;                                    //
;}                                                                            //
;// P8: lxvdsx                                                                //
;// P9: lxvdsx                                                                //
;vector unsigned long long spltMemValull(unsigned long long *ptr) {           //
;  return (vector unsigned long long)*ptr;                                    //
;}                                                                            //
;// P8: constant pool load (possible: vmrgew (xxlxor), (vspltisw))            //
;// P9: constant pool load (possible: vmrgew (xxlxor), (vspltisw))            //
;vector unsigned long long spltCnstConvftoull() {                             //
;  return (vector unsigned long long) 4.74f;                                  //
;}                                                                            //
;// P8: xxmrghd, xvcvdpuxds                                                   //
;// P9: xxmrghd, xvcvdpuxds                                                   //
;vector unsigned long long fromRegsConvftoull(float a, float b) {             //
;  return (vector unsigned long long) { a, b };                               //
;}                                                                            //
;// P8: lxvd2x, xxswapd                                                       //
;// P9: lxvx (even lxv)                                                       //
;vector unsigned long long fromDiffConstsConvftoull() {                       //
;  return (vector unsigned long long) { 24.46f, 234.f };                      //
;}                                                                            //
;// P8: 2 x lxsspx, xxmrghd, xvcvdpuxds                                       //
;// P9: 2 x lxssp, xxmrghd, xvcvdpuxds                                        //
;vector unsigned long long fromDiffMemConsAConvftoull(float *ptr) {           //
;  return (vector unsigned long long) { ptr[0], ptr[1] };                     //
;}                                                                            //
;// P8: 2 x lxsspx, xxmrghd, xvcvdpuxds                                       //
;// P9: 2 x lxssp, xxmrghd, xvcvdpuxds                                        //
;vector unsigned long long fromDiffMemConsDConvftoull(float *ptr) {           //
;  return (vector unsigned long long) { ptr[3], ptr[2] };                     //
;}                                                                            //
;// P8: sldi 2, lfsux, lxsspx, xxmrghd, xvcvdpuxds                            //
;// P9: sldi 2, lfsux, lfs, xxmrghd, xvcvdpuxds                               //
;vector unsigned long long fromDiffMemVarAConvftoull(float *arr, int elem) {  //
;  return (vector unsigned long long) { arr[elem], arr[elem+1] };             //
;}                                                                            //
;// P8: sldi 2, lfsux, lxsspx, xxmrghd, xvcvdpuxds                            //
;// P9: sldi 2, lfsux, lfs, xxmrghd, xvcvdpuxds                               //
;vector unsigned long long fromDiffMemVarDConvftoull(float *arr, int elem) {  //
;  return (vector unsigned long long) { arr[elem], arr[elem-1] };             //
;}                                                                            //
;// P8: xscvdpuxds, xxspltd                                                   //
;// P9: xscvdpuxds, xxspltd                                                   //
;vector unsigned long long spltRegValConvftoull(float val) {                  //
;  return (vector unsigned long long) val;                                    //
;}                                                                            //
;// P8: lxsspx, xscvdpuxds, xxspltd                                           //
;// P9: lfs, xscvdpuxds, xxspltd                                              //
;vector unsigned long long spltMemValConvftoull(float *ptr) {                 //
;  return (vector unsigned long long)*ptr;                                    //
;}                                                                            //
;// P8: constant pool load (possible: vmrgew (xxlxor), (vspltisw))            //
;// P9: constant pool load (possible: vmrgew (xxlxor), (vspltisw))            //
;vector unsigned long long spltCnstConvdtoull() {                             //
;  return (vector unsigned long long) 4.74;                                   //
;}                                                                            //
;// P8: xxmrghd, xvcvdpuxds                                                   //
;// P9: xxmrghd, xvcvdpuxds                                                   //
;vector unsigned long long fromRegsConvdtoull(double a, double b) {           //
;  return (vector unsigned long long) { a, b };                               //
;}                                                                            //
;// P8: lxvd2x, xxswapd                                                       //
;// P9: lxvx (even lxv)                                                       //
;vector unsigned long long fromDiffConstsConvdtoull() {                       //
;  return (vector unsigned long long) { 24.46, 234. };                        //
;}                                                                            //
;// P8: lxvd2x, xxswapd, xvcvdpuxds                                           //
;// P9: lxvx, xvcvdpuxds                                                      //
;vector unsigned long long fromDiffMemConsAConvdtoull(double *ptr) {          //
;  return (vector unsigned long long) { ptr[0], ptr[1] };                     //
;}                                                                            //
;// P8: lxvd2x, xvcvdpuxds                                                    //
;// P9: lxvx, xxswapd, xvcvdpuxds                                             //
;vector unsigned long long fromDiffMemConsDConvdtoull(double *ptr) {          //
;  return (vector unsigned long long) { ptr[3], ptr[2] };                     //
;}                                                                            //
;// P8: sldi 3, lxvd2x, xxswapd, xvcvdpuxds                                   //
;// P9: sldi 3, lxvx, xvcvdpuxds                                              //
;vector unsigned long long fromDiffMemVarAConvdtoull(double *arr, int elem) { //
;  return (vector unsigned long long) { arr[elem], arr[elem+1] };             //
;}                                                                            //
;// P8: sldi 3, lxvd2x, xvcvdpuxds                                            //
;// P9: sldi 3, lxvx, xxswapd, xvcvdpuxds                                     //
;vector unsigned long long fromDiffMemVarDConvdtoull(double *arr, int elem) { //
;  return (vector unsigned long long) { arr[elem], arr[elem-1] };             //
;}                                                                            //
;// P8: xscvdpuxds, xxspltd                                                   //
;// P9: xscvdpuxds, xxspltd                                                   //
;vector unsigned long long spltRegValConvdtoull(double val) {                 //
;  return (vector unsigned long long) val;                                    //
;}                                                                            //
;// P8: lxvdsx, xvcvdpuxds                                                    //
;// P9: lxvdsx, xvcvdpuxds                                                    //
;vector unsigned long long spltMemValConvdtoull(double *ptr) {                //
;  return (vector unsigned long long)*ptr;                                    //
;}                                                                            //
;/*========================== unsigned long long ==============================*/

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @allZeroi() {
entry:
  ret <4 x i32> zeroinitializer
; P9BE-LABEL: allZeroi
; P9LE-LABEL: allZeroi
; P8BE-LABEL: allZeroi
; P8LE-LABEL: allZeroi
; P9BE: xxlxor v2, v2, v2
; P9BE: blr
; P9LE: xxlxor v2, v2, v2
; P9LE: blr
; P8BE: xxlxor v2, v2, v2
; P8BE: blr
; P8LE: xxlxor v2, v2, v2
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @allOnei() {
entry:
  ret <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
; P9BE-LABEL: allOnei
; P9LE-LABEL: allOnei
; P8BE-LABEL: allOnei
; P8LE-LABEL: allOnei
; P9BE: xxspltib v2, 255
; P9BE: blr
; P9LE: xxspltib v2, 255
; P9LE: blr
; P8BE: vspltisb v2, -1
; P8BE: blr
; P8LE: vspltisb v2, -1
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @spltConst1i() {
entry:
  ret <4 x i32> <i32 1, i32 1, i32 1, i32 1>
; P9BE-LABEL: spltConst1i
; P9LE-LABEL: spltConst1i
; P8BE-LABEL: spltConst1i
; P8LE-LABEL: spltConst1i
; P9BE: vspltisw v2, 1
; P9BE: blr
; P9LE: vspltisw v2, 1
; P9LE: blr
; P8BE: vspltisw v2, 1
; P8BE: blr
; P8LE: vspltisw v2, 1
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @spltConst16ki() {
entry:
  ret <4 x i32> <i32 32767, i32 32767, i32 32767, i32 32767>
; P9BE-LABEL: spltConst16ki
; P9LE-LABEL: spltConst16ki
; P8BE-LABEL: spltConst16ki
; P8LE-LABEL: spltConst16ki
; P9BE: vspltisw v2, -15
; P9BE: vsrw v2, v2, v2
; P9BE: blr
; P9LE: vspltisw v2, -15
; P9LE: vsrw v2, v2, v2
; P9LE: blr
; P8BE: vspltisw v2, -15
; P8BE: vsrw v2, v2, v2
; P8BE: blr
; P8LE: vspltisw v2, -15
; P8LE: vsrw v2, v2, v2
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @spltConst32ki() {
entry:
  ret <4 x i32> <i32 65535, i32 65535, i32 65535, i32 65535>
; P9BE-LABEL: spltConst32ki
; P9LE-LABEL: spltConst32ki
; P8BE-LABEL: spltConst32ki
; P8LE-LABEL: spltConst32ki
; P9BE: vspltisw v2, -16
; P9BE: vsrw v2, v2, v2
; P9BE: blr
; P9LE: vspltisw v2, -16
; P9LE: vsrw v2, v2, v2
; P9LE: blr
; P8BE: vspltisw v2, -16
; P8BE: vsrw v2, v2, v2
; P8BE: blr
; P8LE: vspltisw v2, -16
; P8LE: vsrw v2, v2, v2
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @fromRegsi(i32 signext %a, i32 signext %b, i32 signext %c, i32 signext %d) {
entry:
  %vecinit = insertelement <4 x i32> undef, i32 %a, i32 0
  %vecinit1 = insertelement <4 x i32> %vecinit, i32 %b, i32 1
  %vecinit2 = insertelement <4 x i32> %vecinit1, i32 %c, i32 2
  %vecinit3 = insertelement <4 x i32> %vecinit2, i32 %d, i32 3
  ret <4 x i32> %vecinit3
; P9BE-LABEL: fromRegsi
; P9LE-LABEL: fromRegsi
; P8BE-LABEL: fromRegsi
; P8LE-LABEL: fromRegsi
; P9BE-DAG: mtvsrdd [[REG1:v[0-9]+]], r3, r5
; P9BE-DAG: mtvsrdd [[REG2:v[0-9]+]], r4, r6
; P9BE: vmrgow v2, [[REG1]], [[REG2]]
; P9BE: blr
; P9LE-DAG: mtvsrdd [[REG1:v[0-9]+]], r5, r3
; P9LE-DAG: mtvsrdd [[REG2:v[0-9]+]], r6, r4
; P9LE: vmrgow v2, [[REG2]], [[REG1]]
; P9LE: blr
; P8BE-DAG: mtvsrwz {{[vf]}}[[REG1:[0-9]+]], r3
; P8BE-DAG: mtvsrwz {{[vf]}}[[REG2:[0-9]+]], r4
; P8BE-DAG: mtvsrwz {{[vf]}}[[REG3:[0-9]+]], r5
; P8BE-DAG: mtvsrwz {{[vf]}}[[REG4:[0-9]+]], r6
; P8BE-DAG: xxmrghd [[REG5:v[0-9]+]], {{[v][s]*}}[[REG1]], {{[v][s]*}}[[REG3]]
; P8BE-DAG: xxmrghd [[REG6:v[0-9]+]], {{[v][s]*}}[[REG2]], {{[v][s]*}}[[REG4]]
; P8BE: vmrgow v2, [[REG5]], [[REG6]]
; P8LE-DAG: mtvsrwz {{[vf]}}[[REG1:[0-9]+]], r3
; P8LE-DAG: mtvsrwz {{[vf]}}[[REG2:[0-9]+]], r4
; P8LE-DAG: mtvsrwz {{[vf]}}[[REG3:[0-9]+]], r5
; P8LE-DAG: mtvsrwz {{[vf]}}[[REG4:[0-9]+]], r6
; P8LE: xxmrghd [[REG5:v[0-9]+]], {{[v][s]*}}[[REG3]], {{[v][s]*}}[[REG1]]
; P8LE: xxmrghd [[REG6:v[0-9]+]], {{[v][s]*}}[[REG4]], {{[v][s]*}}[[REG2]]
; P8LE: vmrgow v2, [[REG6]], [[REG5]]
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @fromDiffConstsi() {
entry:
  ret <4 x i32> <i32 242, i32 -113, i32 889, i32 19>
; P9BE-LABEL: fromDiffConstsi
; P9LE-LABEL: fromDiffConstsi
; P8BE-LABEL: fromDiffConstsi
; P8LE-LABEL: fromDiffConstsi
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvw4x
; P8BE: blr
; P8LE: lvx
; P8LE-NOT: xxswapd
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemConsAi(i32* nocapture readonly %arr) {
entry:
  %0 = load i32, i32* %arr, align 4
  %vecinit = insertelement <4 x i32> undef, i32 %0, i32 0
  %arrayidx1 = getelementptr inbounds i32, i32* %arr, i64 1
  %1 = load i32, i32* %arrayidx1, align 4
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %1, i32 1
  %arrayidx3 = getelementptr inbounds i32, i32* %arr, i64 2
  %2 = load i32, i32* %arrayidx3, align 4
  %vecinit4 = insertelement <4 x i32> %vecinit2, i32 %2, i32 2
  %arrayidx5 = getelementptr inbounds i32, i32* %arr, i64 3
  %3 = load i32, i32* %arrayidx5, align 4
  %vecinit6 = insertelement <4 x i32> %vecinit4, i32 %3, i32 3
  ret <4 x i32> %vecinit6
; P9BE-LABEL: fromDiffMemConsAi
; P9LE-LABEL: fromDiffMemConsAi
; P8BE-LABEL: fromDiffMemConsAi
; P8LE-LABEL: fromDiffMemConsAi
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvw4x
; P8BE: blr
; P8LE: lxvd2x
; P8LE: xxswapd
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemConsDi(i32* nocapture readonly %arr) {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 3
  %0 = load i32, i32* %arrayidx, align 4
  %vecinit = insertelement <4 x i32> undef, i32 %0, i32 0
  %arrayidx1 = getelementptr inbounds i32, i32* %arr, i64 2
  %1 = load i32, i32* %arrayidx1, align 4
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %1, i32 1
  %arrayidx3 = getelementptr inbounds i32, i32* %arr, i64 1
  %2 = load i32, i32* %arrayidx3, align 4
  %vecinit4 = insertelement <4 x i32> %vecinit2, i32 %2, i32 2
  %3 = load i32, i32* %arr, align 4
  %vecinit6 = insertelement <4 x i32> %vecinit4, i32 %3, i32 3
  ret <4 x i32> %vecinit6
; P9BE-LABEL: fromDiffMemConsDi
; P9LE-LABEL: fromDiffMemConsDi
; P8BE-LABEL: fromDiffMemConsDi
; P8LE-LABEL: fromDiffMemConsDi
; P9BE: lxv
; P9BE: lxv
; P9BE: vperm
; P9BE: blr
; P9LE: lxv
; P9LE: lxv
; P9LE: vperm
; P9LE: blr
; P8BE: lxvw4x
; P8BE: lxvw4x
; P8BE: vperm
; P8BE: blr
; P8LE: lxvd2x
; P8LE-DAG: lvx
; P8LE: xxswapd
; P8LE: vperm
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemVarAi(i32* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %vecinit = insertelement <4 x i32> undef, i32 %0, i32 0
  %add = add nsw i32 %elem, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %arr, i64 %idxprom1
  %1 = load i32, i32* %arrayidx2, align 4
  %vecinit3 = insertelement <4 x i32> %vecinit, i32 %1, i32 1
  %add4 = add nsw i32 %elem, 2
  %idxprom5 = sext i32 %add4 to i64
  %arrayidx6 = getelementptr inbounds i32, i32* %arr, i64 %idxprom5
  %2 = load i32, i32* %arrayidx6, align 4
  %vecinit7 = insertelement <4 x i32> %vecinit3, i32 %2, i32 2
  %add8 = add nsw i32 %elem, 3
  %idxprom9 = sext i32 %add8 to i64
  %arrayidx10 = getelementptr inbounds i32, i32* %arr, i64 %idxprom9
  %3 = load i32, i32* %arrayidx10, align 4
  %vecinit11 = insertelement <4 x i32> %vecinit7, i32 %3, i32 3
  ret <4 x i32> %vecinit11
; P9BE-LABEL: fromDiffMemVarAi
; P9LE-LABEL: fromDiffMemVarAi
; P8BE-LABEL: fromDiffMemVarAi
; P8LE-LABEL: fromDiffMemVarAi
; P9BE: sldi r4, r4, 2
; P9BE: lxvx v2, r3, r4
; P9BE: blr
; P9LE: sldi r4, r4, 2
; P9LE: lxvx v2, r3, r4
; P9LE: blr
; P8BE: sldi r4, r4, 2
; P8BE: lxvw4x {{[vs0-9]+}}, r3, r4
; P8BE: blr
; P8LE: sldi r4, r4, 2
; P8LE: lxvd2x {{[vs0-9]+}}, r3, r4
; P8LE: xxswapd
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemVarDi(i32* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %vecinit = insertelement <4 x i32> undef, i32 %0, i32 0
  %sub = add nsw i32 %elem, -1
  %idxprom1 = sext i32 %sub to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %arr, i64 %idxprom1
  %1 = load i32, i32* %arrayidx2, align 4
  %vecinit3 = insertelement <4 x i32> %vecinit, i32 %1, i32 1
  %sub4 = add nsw i32 %elem, -2
  %idxprom5 = sext i32 %sub4 to i64
  %arrayidx6 = getelementptr inbounds i32, i32* %arr, i64 %idxprom5
  %2 = load i32, i32* %arrayidx6, align 4
  %vecinit7 = insertelement <4 x i32> %vecinit3, i32 %2, i32 2
  %sub8 = add nsw i32 %elem, -3
  %idxprom9 = sext i32 %sub8 to i64
  %arrayidx10 = getelementptr inbounds i32, i32* %arr, i64 %idxprom9
  %3 = load i32, i32* %arrayidx10, align 4
  %vecinit11 = insertelement <4 x i32> %vecinit7, i32 %3, i32 3
  ret <4 x i32> %vecinit11
; P9BE-LABEL: fromDiffMemVarDi
; P9LE-LABEL: fromDiffMemVarDi
; P8BE-LABEL: fromDiffMemVarDi
; P8LE-LABEL: fromDiffMemVarDi
; P9BE: sldi {{r[0-9]+}}, r4, 2
; P9BE-DAG: lxvx {{v[0-9]+}}
; P9BE-DAG: lxvx
; P9BE: vperm
; P9BE: blr
; P9LE: sldi {{r[0-9]+}}, r4, 2
; P9LE-DAG: lxvx {{v[0-9]+}}
; P9LE-DAG: lxvx
; P9LE: vperm
; P9LE: blr
; P8BE: sldi {{r[0-9]+}}, r4, 2
; P8BE-DAG: lxvw4x {{v[0-9]+}}, 0, r3
; P8BE-DAG: lxvw4x
; P8BE: vperm
; P8BE: blr
; P8LE: sldi {{r[0-9]+}}, r4, 2
; P8LE-DAG: lxvd2x
; P8LE-DAG: lxvd2x
; P8LE: xxswapd
; P8LE: vperm
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromRandMemConsi(i32* nocapture readonly %arr) {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 4
  %0 = load i32, i32* %arrayidx, align 4
  %vecinit = insertelement <4 x i32> undef, i32 %0, i32 0
  %arrayidx1 = getelementptr inbounds i32, i32* %arr, i64 18
  %1 = load i32, i32* %arrayidx1, align 4
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %1, i32 1
  %arrayidx3 = getelementptr inbounds i32, i32* %arr, i64 2
  %2 = load i32, i32* %arrayidx3, align 4
  %vecinit4 = insertelement <4 x i32> %vecinit2, i32 %2, i32 2
  %arrayidx5 = getelementptr inbounds i32, i32* %arr, i64 88
  %3 = load i32, i32* %arrayidx5, align 4
  %vecinit6 = insertelement <4 x i32> %vecinit4, i32 %3, i32 3
  ret <4 x i32> %vecinit6
; P9BE-LABEL: fromRandMemConsi
; P9LE-LABEL: fromRandMemConsi
; P8BE-LABEL: fromRandMemConsi
; P8LE-LABEL: fromRandMemConsi
; P9BE: lwz
; P9BE: lwz
; P9BE: lwz
; P9BE: lwz
; P9BE: mtvsrdd
; P9BE: mtvsrdd
; P9BE: vmrgow
; P9LE: lwz
; P9LE: lwz
; P9LE: lwz
; P9LE: lwz
; P9LE: mtvsrdd
; P9LE: mtvsrdd
; P9LE: vmrgow
; P8BE: lwz
; P8BE: lwz
; P8BE: lwz
; P8BE: lwz
; P8BE: mtvsrwz
; P8BE: mtvsrwz
; P8BE: mtvsrwz
; P8BE: mtvsrwz
; P8BE: xxmrghd
; P8BE: xxmrghd
; P8BE: vmrgow
; P8LE: lwz
; P8LE: lwz
; P8LE: lwz
; P8LE: lwz
; P8LE: mtvsrwz
; P8LE: mtvsrwz
; P8LE: mtvsrwz
; P8LE: mtvsrwz
; P8LE: xxmrghd
; P8LE: xxmrghd
; P8LE: vmrgow
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromRandMemVari(i32* nocapture readonly %arr, i32 signext %elem) {
entry:
  %add = add nsw i32 %elem, 4
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %vecinit = insertelement <4 x i32> undef, i32 %0, i32 0
  %add1 = add nsw i32 %elem, 1
  %idxprom2 = sext i32 %add1 to i64
  %arrayidx3 = getelementptr inbounds i32, i32* %arr, i64 %idxprom2
  %1 = load i32, i32* %arrayidx3, align 4
  %vecinit4 = insertelement <4 x i32> %vecinit, i32 %1, i32 1
  %add5 = add nsw i32 %elem, 2
  %idxprom6 = sext i32 %add5 to i64
  %arrayidx7 = getelementptr inbounds i32, i32* %arr, i64 %idxprom6
  %2 = load i32, i32* %arrayidx7, align 4
  %vecinit8 = insertelement <4 x i32> %vecinit4, i32 %2, i32 2
  %add9 = add nsw i32 %elem, 8
  %idxprom10 = sext i32 %add9 to i64
  %arrayidx11 = getelementptr inbounds i32, i32* %arr, i64 %idxprom10
  %3 = load i32, i32* %arrayidx11, align 4
  %vecinit12 = insertelement <4 x i32> %vecinit8, i32 %3, i32 3
  ret <4 x i32> %vecinit12
; P9BE-LABEL: fromRandMemVari
; P9LE-LABEL: fromRandMemVari
; P8BE-LABEL: fromRandMemVari
; P8LE-LABEL: fromRandMemVari
; P9BE: sldi r4, r4, 2
; P9BE: lwz
; P9BE: lwz
; P9BE: lwz
; P9BE: lwz
; P9BE: mtvsrdd
; P9BE: mtvsrdd
; P9BE: vmrgow
; P9LE: sldi r4, r4, 2
; P9LE: lwz
; P9LE: lwz
; P9LE: lwz
; P9LE: lwz
; P9LE: mtvsrdd
; P9LE: mtvsrdd
; P9LE: vmrgow
; P8BE: sldi r4, r4, 2
; P8BE: lwz
; P8BE: lwz
; P8BE: lwz
; P8BE: lwz
; P8BE: mtvsrwz
; P8BE: mtvsrwz
; P8BE: mtvsrwz
; P8BE: mtvsrwz
; P8BE: xxmrghd
; P8BE: xxmrghd
; P8BE: vmrgow
; P8LE: sldi r4, r4, 2
; P8LE: lwz
; P8LE: lwz
; P8LE: lwz
; P8LE: lwz
; P8LE: mtvsrwz
; P8LE: mtvsrwz
; P8LE: mtvsrwz
; P8LE: mtvsrwz
; P8LE: xxmrghd
; P8LE: xxmrghd
; P8LE: vmrgow
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @spltRegVali(i32 signext %val) {
entry:
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %val, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; P9BE-LABEL: spltRegVali
; P9LE-LABEL: spltRegVali
; P8BE-LABEL: spltRegVali
; P8LE-LABEL: spltRegVali
; P9BE: mtvsrws v2, r3
; P9BE: blr
; P9LE: mtvsrws v2, r3
; P9LE: blr
; P8BE: mtvsrwz {{[vsf0-9]+}}, r3
; P8BE: xxspltw v2, {{[vsf0-9]+}}, 1
; P8BE: blr
; P8LE: mtvsrwz {{[vsf0-9]+}}, r3
; P8LE: xxspltw v2, {{[vsf0-9]+}}, 1
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @spltMemVali(i32* nocapture readonly %ptr) {
entry:
  %0 = load i32, i32* %ptr, align 4
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %0, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; P9BE-LABEL: spltMemVali
; P9LE-LABEL: spltMemVali
; P8BE-LABEL: spltMemVali
; P8LE-LABEL: spltMemVali
; P9BE: lxvwsx v2, 0, r3
; P9BE: blr
; P9LE: lxvwsx v2, 0, r3
; P9LE: blr
; P8BE: lxsiwax {{[vsf0-9]+}}, 0, r3
; P8BE: xxspltw v2, {{[vsf0-9]+}}, 1
; P8BE: blr
; P8LE: lxsiwax {{[vsf0-9]+}}, 0, r3
; P8LE: xxspltw v2, {{[vsf0-9]+}}, 1
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @spltCnstConvftoi() {
entry:
  ret <4 x i32> <i32 4, i32 4, i32 4, i32 4>
; P9BE-LABEL: spltCnstConvftoi
; P9LE-LABEL: spltCnstConvftoi
; P8BE-LABEL: spltCnstConvftoi
; P8LE-LABEL: spltCnstConvftoi
; P9BE: vspltisw v2, 4
; P9BE: blr
; P9LE: vspltisw v2, 4
; P9LE: blr
; P8BE: vspltisw v2, 4
; P8BE: blr
; P8LE: vspltisw v2, 4
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @fromRegsConvftoi(float %a, float %b, float %c, float %d) {
entry:
  %conv = fptosi float %a to i32
  %vecinit = insertelement <4 x i32> undef, i32 %conv, i32 0
  %conv1 = fptosi float %b to i32
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %conv1, i32 1
  %conv3 = fptosi float %c to i32
  %vecinit4 = insertelement <4 x i32> %vecinit2, i32 %conv3, i32 2
  %conv5 = fptosi float %d to i32
  %vecinit6 = insertelement <4 x i32> %vecinit4, i32 %conv5, i32 3
  ret <4 x i32> %vecinit6
; P9BE-LABEL: fromRegsConvftoi
; P9LE-LABEL: fromRegsConvftoi
; P8BE-LABEL: fromRegsConvftoi
; P8LE-LABEL: fromRegsConvftoi
; P9BE-DAG: xxmrghd {{[vs]+}}[[REG1:[0-9]+]], vs1, vs3
; P9BE-DAG: xxmrghd {{[vs]+}}[[REG2:[0-9]+]], vs2, vs4
; P9BE-DAG: xvcvdpsp [[REG3:v[0-9]+]], {{[vs]+}}[[REG1]]
; P9BE-DAG: xvcvdpsp [[REG4:v[0-9]+]], {{[vs]+}}[[REG2]]
; P9BE: vmrgew v2, [[REG3]], [[REG4]]
; P9BE: xvcvspsxws v2, v2
; P9LE-DAG: xxmrghd {{[vs]+}}[[REG1:[0-9]+]], vs3, vs1
; P9LE-DAG: xxmrghd {{[vs]+}}[[REG2:[0-9]+]], vs4, vs2
; P9LE-DAG: xvcvdpsp [[REG3:v[0-9]+]], {{[vs]+}}[[REG1]]
; P9LE-DAG: xvcvdpsp [[REG4:v[0-9]+]], {{[vs]+}}[[REG2]]
; P9LE: vmrgew v2, [[REG4]], [[REG3]]
; P9LE: xvcvspsxws v2, v2
; P8BE-DAG: xxmrghd {{[vs]+}}[[REG1:[0-9]+]], vs1, vs3
; P8BE-DAG: xxmrghd {{[vs]+}}[[REG2:[0-9]+]], vs2, vs4
; P8BE-DAG: xvcvdpsp [[REG3:v[0-9]+]], {{[vs]+}}[[REG1]]
; P8BE-DAG: xvcvdpsp [[REG4:v[0-9]+]], {{[vs]+}}[[REG2]]
; P8BE: vmrgew v2, [[REG3]], [[REG4]]
; P8BE: xvcvspsxws v2, v2
; P8LE-DAG: xxmrghd {{[vs]+}}[[REG1:[0-9]+]], vs3, vs1
; P8LE-DAG: xxmrghd {{[vs]+}}[[REG2:[0-9]+]], vs4, vs2
; P8LE-DAG: xvcvdpsp [[REG3:v[0-9]+]], {{[vs]+}}[[REG1]]
; P8LE-DAG: xvcvdpsp [[REG4:v[0-9]+]], {{[vs]+}}[[REG2]]
; P8LE: vmrgew v2, [[REG4]], [[REG3]]
; P8LE: xvcvspsxws v2, v2
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @fromDiffConstsConvftoi() {
entry:
  ret <4 x i32> <i32 24, i32 234, i32 988, i32 422>
; P9BE-LABEL: fromDiffConstsConvftoi
; P9LE-LABEL: fromDiffConstsConvftoi
; P8BE-LABEL: fromDiffConstsConvftoi
; P8LE-LABEL: fromDiffConstsConvftoi
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvw4x
; P8BE: blr
; P8LE: lvx
; P8LE-NOT: xxswapd
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemConsAConvftoi(float* nocapture readonly %ptr) {
entry:
  %0 = bitcast float* %ptr to <4 x float>*
  %1 = load <4 x float>, <4 x float>* %0, align 4
  %2 = fptosi <4 x float> %1 to <4 x i32>
  ret <4 x i32> %2
; P9BE-LABEL: fromDiffMemConsAConvftoi
; P9LE-LABEL: fromDiffMemConsAConvftoi
; P8BE-LABEL: fromDiffMemConsAConvftoi
; P8LE-LABEL: fromDiffMemConsAConvftoi
; P9BE: lxv [[REG1:[vs0-9]+]], 0(r3)
; P9BE: xvcvspsxws v2, [[REG1]]
; P9BE: blr
; P9LE: lxv [[REG1:[vs0-9]+]], 0(r3)
; P9LE: xvcvspsxws v2, [[REG1]]
; P9LE: blr
; P8BE: lxvw4x [[REG1:[vs0-9]+]], 0, r3
; P8BE: xvcvspsxws v2, [[REG1]]
; P8BE: blr
; P8LE: lxvd2x [[REG1:[vs0-9]+]], 0, r3
; P8LE: xxswapd
; P8LE: xvcvspsxws v2, v2
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemConsDConvftoi(float* nocapture readonly %ptr) {
entry:
  %arrayidx = getelementptr inbounds float, float* %ptr, i64 3
  %0 = load float, float* %arrayidx, align 4
  %conv = fptosi float %0 to i32
  %vecinit = insertelement <4 x i32> undef, i32 %conv, i32 0
  %arrayidx1 = getelementptr inbounds float, float* %ptr, i64 2
  %1 = load float, float* %arrayidx1, align 4
  %conv2 = fptosi float %1 to i32
  %vecinit3 = insertelement <4 x i32> %vecinit, i32 %conv2, i32 1
  %arrayidx4 = getelementptr inbounds float, float* %ptr, i64 1
  %2 = load float, float* %arrayidx4, align 4
  %conv5 = fptosi float %2 to i32
  %vecinit6 = insertelement <4 x i32> %vecinit3, i32 %conv5, i32 2
  %3 = load float, float* %ptr, align 4
  %conv8 = fptosi float %3 to i32
  %vecinit9 = insertelement <4 x i32> %vecinit6, i32 %conv8, i32 3
  ret <4 x i32> %vecinit9
; P9BE-LABEL: fromDiffMemConsDConvftoi
; P9LE-LABEL: fromDiffMemConsDConvftoi
; P8BE-LABEL: fromDiffMemConsDConvftoi
; P8LE-LABEL: fromDiffMemConsDConvftoi
; P9BE: lxv
; P9BE: lxv
; P9BE: vperm
; P9BE: xvcvspsxws
; P9BE: blr
; P9LE: lxv
; P9LE: lxv
; P9LE: vperm
; P9LE: xvcvspsxws
; P9LE: blr
; P8BE: lxvw4x
; P8BE: lxvw4x
; P8BE: vperm
; P8BE: xvcvspsxws
; P8BE: blr
; P8LE: lxvd2x
; P8LE-DAG: lvx
; P8LE: xxswapd
; P8LE: vperm
; P8LE: xvcvspsxws
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemVarAConvftoi(float* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds float, float* %arr, i64 %idxprom
  %0 = load float, float* %arrayidx, align 4
  %conv = fptosi float %0 to i32
  %vecinit = insertelement <4 x i32> undef, i32 %conv, i32 0
  %add = add nsw i32 %elem, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds float, float* %arr, i64 %idxprom1
  %1 = load float, float* %arrayidx2, align 4
  %conv3 = fptosi float %1 to i32
  %vecinit4 = insertelement <4 x i32> %vecinit, i32 %conv3, i32 1
  %add5 = add nsw i32 %elem, 2
  %idxprom6 = sext i32 %add5 to i64
  %arrayidx7 = getelementptr inbounds float, float* %arr, i64 %idxprom6
  %2 = load float, float* %arrayidx7, align 4
  %conv8 = fptosi float %2 to i32
  %vecinit9 = insertelement <4 x i32> %vecinit4, i32 %conv8, i32 2
  %add10 = add nsw i32 %elem, 3
  %idxprom11 = sext i32 %add10 to i64
  %arrayidx12 = getelementptr inbounds float, float* %arr, i64 %idxprom11
  %3 = load float, float* %arrayidx12, align 4
  %conv13 = fptosi float %3 to i32
  %vecinit14 = insertelement <4 x i32> %vecinit9, i32 %conv13, i32 3
  ret <4 x i32> %vecinit14
; P9BE-LABEL: fromDiffMemVarAConvftoi
; P9LE-LABEL: fromDiffMemVarAConvftoi
; P8BE-LABEL: fromDiffMemVarAConvftoi
; P8LE-LABEL: fromDiffMemVarAConvftoi
; FIXME: implement finding consecutive loads with pre-inc
; P9BE: lfsux
; P9LE: lfsux
; P8BE: lfsux
; P8LE: lfsux
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemVarDConvftoi(float* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds float, float* %arr, i64 %idxprom
  %0 = load float, float* %arrayidx, align 4
  %conv = fptosi float %0 to i32
  %vecinit = insertelement <4 x i32> undef, i32 %conv, i32 0
  %sub = add nsw i32 %elem, -1
  %idxprom1 = sext i32 %sub to i64
  %arrayidx2 = getelementptr inbounds float, float* %arr, i64 %idxprom1
  %1 = load float, float* %arrayidx2, align 4
  %conv3 = fptosi float %1 to i32
  %vecinit4 = insertelement <4 x i32> %vecinit, i32 %conv3, i32 1
  %sub5 = add nsw i32 %elem, -2
  %idxprom6 = sext i32 %sub5 to i64
  %arrayidx7 = getelementptr inbounds float, float* %arr, i64 %idxprom6
  %2 = load float, float* %arrayidx7, align 4
  %conv8 = fptosi float %2 to i32
  %vecinit9 = insertelement <4 x i32> %vecinit4, i32 %conv8, i32 2
  %sub10 = add nsw i32 %elem, -3
  %idxprom11 = sext i32 %sub10 to i64
  %arrayidx12 = getelementptr inbounds float, float* %arr, i64 %idxprom11
  %3 = load float, float* %arrayidx12, align 4
  %conv13 = fptosi float %3 to i32
  %vecinit14 = insertelement <4 x i32> %vecinit9, i32 %conv13, i32 3
  ret <4 x i32> %vecinit14
; P9BE-LABEL: fromDiffMemVarDConvftoi
; P9LE-LABEL: fromDiffMemVarDConvftoi
; P8BE-LABEL: fromDiffMemVarDConvftoi
; P8LE-LABEL: fromDiffMemVarDConvftoi
; FIXME: implement finding consecutive loads with pre-inc
; P9BE: lfsux
; P9LE: lfsux
; P8BE: lfsux
; P8LE: lfsux
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @spltRegValConvftoi(float %val) {
entry:
  %conv = fptosi float %val to i32
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %conv, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; P9BE-LABEL: spltRegValConvftoi
; P9LE-LABEL: spltRegValConvftoi
; P8BE-LABEL: spltRegValConvftoi
; P8LE-LABEL: spltRegValConvftoi
; P9BE: xscvdpsxws f[[REG1:[0-9]+]], f1
; P9BE: xxspltw v2, vs[[REG1]], 1
; P9BE: blr
; P9LE: xscvdpsxws f[[REG1:[0-9]+]], f1
; P9LE: xxspltw v2, vs[[REG1]], 1
; P9LE: blr
; P8BE: xscvdpsxws f[[REG1:[0-9]+]], f1
; P8BE: xxspltw v2, vs[[REG1]], 1
; P8BE: blr
; P8LE: xscvdpsxws f[[REG1:[0-9]+]], f1
; P8LE: xxspltw v2, vs[[REG1]], 1
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @spltMemValConvftoi(float* nocapture readonly %ptr) {
entry:
  %0 = load float, float* %ptr, align 4
  %conv = fptosi float %0 to i32
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %conv, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; P9BE-LABEL: spltMemValConvftoi
; P9LE-LABEL: spltMemValConvftoi
; P8BE-LABEL: spltMemValConvftoi
; P8LE-LABEL: spltMemValConvftoi
; P9BE: lxvwsx [[REG1:[vs0-9]+]], 0, r3
; P9BE: xvcvspsxws v2, [[REG1]]
; P9LE: [[REG1:[vs0-9]+]], 0, r3
; P9LE: xvcvspsxws v2, [[REG1]]
; P8BE: lfsx [[REG1:f[0-9]+]], 0, r3
; P8BE: xscvdpsxws f[[REG2:[0-9]+]], [[REG1]]
; P8BE: xxspltw v2, vs[[REG2]], 1
; P8LE: lfsx [[REG1:f[0-9]+]], 0, r3
; P8LE: xscvdpsxws f[[REG2:[vs0-9]+]], [[REG1]]
; P8LE: xxspltw v2, vs[[REG2]], 1
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @spltCnstConvdtoi() {
entry:
  ret <4 x i32> <i32 4, i32 4, i32 4, i32 4>
; P9BE-LABEL: spltCnstConvdtoi
; P9LE-LABEL: spltCnstConvdtoi
; P8BE-LABEL: spltCnstConvdtoi
; P8LE-LABEL: spltCnstConvdtoi
; P9BE: vspltisw v2, 4
; P9BE: blr
; P9LE: vspltisw v2, 4
; P9LE: blr
; P8BE: vspltisw v2, 4
; P8BE: blr
; P8LE: vspltisw v2, 4
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @fromRegsConvdtoi(double %a, double %b, double %c, double %d) {
entry:
  %conv = fptosi double %a to i32
  %vecinit = insertelement <4 x i32> undef, i32 %conv, i32 0
  %conv1 = fptosi double %b to i32
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %conv1, i32 1
  %conv3 = fptosi double %c to i32
  %vecinit4 = insertelement <4 x i32> %vecinit2, i32 %conv3, i32 2
  %conv5 = fptosi double %d to i32
  %vecinit6 = insertelement <4 x i32> %vecinit4, i32 %conv5, i32 3
  ret <4 x i32> %vecinit6
; P9BE-LABEL: fromRegsConvdtoi
; P9LE-LABEL: fromRegsConvdtoi
; P8BE-LABEL: fromRegsConvdtoi
; P8LE-LABEL: fromRegsConvdtoi
; P9BE-DAG: xxmrghd {{[vs]+}}[[REG1:[0-9]+]], vs1, vs3
; P9BE-DAG: xxmrghd {{[vs]+}}[[REG2:[0-9]+]], vs2, vs4
; P9BE-DAG: xvcvdpsp [[REG3:v[0-9]+]], {{[vs]+}}[[REG1]]
; P9BE-DAG: xvcvdpsp [[REG4:v[0-9]+]], {{[vs]+}}[[REG2]]
; P9BE: vmrgew v2, [[REG3]], [[REG4]]
; P9BE: xvcvspsxws v2, v2
; P9LE-DAG: xxmrghd {{[vs]+}}[[REG1:[0-9]+]], vs3, vs1
; P9LE-DAG: xxmrghd {{[vs]+}}[[REG2:[0-9]+]], vs4, vs2
; P9LE-DAG: xvcvdpsp [[REG3:v[0-9]+]], {{[vs]+}}[[REG1]]
; P9LE-DAG: xvcvdpsp [[REG4:v[0-9]+]], {{[vs]+}}[[REG2]]
; P9LE: vmrgew v2, [[REG4]], [[REG3]]
; P9LE: xvcvspsxws v2, v2
; P8BE-DAG: xxmrghd {{[vs]+}}[[REG1:[0-9]+]], vs1, vs3
; P8BE-DAG: xxmrghd {{[vs]+}}[[REG2:[0-9]+]], vs2, vs4
; P8BE-DAG: xvcvdpsp [[REG3:v[0-9]+]], {{[vs]+}}[[REG1]]
; P8BE-DAG: xvcvdpsp [[REG4:v[0-9]+]], {{[vs]+}}[[REG2]]
; P8BE: vmrgew v2, [[REG3]], [[REG4]]
; P8BE: xvcvspsxws v2, v2
; P8LE-DAG: xxmrghd {{[vs]+}}[[REG1:[0-9]+]], vs3, vs1
; P8LE-DAG: xxmrghd {{[vs]+}}[[REG2:[0-9]+]], vs4, vs2
; P8LE-DAG: xvcvdpsp [[REG3:v[0-9]+]], {{[vs]+}}[[REG1]]
; P8LE-DAG: xvcvdpsp [[REG4:v[0-9]+]], {{[vs]+}}[[REG2]]
; P8LE: vmrgew v2, [[REG4]], [[REG3]]
; P8LE: xvcvspsxws v2, v2
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @fromDiffConstsConvdtoi() {
entry:
  ret <4 x i32> <i32 24, i32 234, i32 988, i32 422>
; P9BE-LABEL: fromDiffConstsConvdtoi
; P9LE-LABEL: fromDiffConstsConvdtoi
; P8BE-LABEL: fromDiffConstsConvdtoi
; P8LE-LABEL: fromDiffConstsConvdtoi
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvw4x
; P8BE: blr
; P8LE: lvx
; P8LE-NOT: xxswapd
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemConsAConvdtoi(double* nocapture readonly %ptr) {
entry:
  %0 = bitcast double* %ptr to <2 x double>*
  %1 = load <2 x double>, <2 x double>* %0, align 8
  %2 = fptosi <2 x double> %1 to <2 x i32>
  %arrayidx4 = getelementptr inbounds double, double* %ptr, i64 2
  %3 = bitcast double* %arrayidx4 to <2 x double>*
  %4 = load <2 x double>, <2 x double>* %3, align 8
  %5 = fptosi <2 x double> %4 to <2 x i32>
  %vecinit9 = shufflevector <2 x i32> %2, <2 x i32> %5, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %vecinit9
; P9BE-LABEL: fromDiffMemConsAConvdtoi
; P9LE-LABEL: fromDiffMemConsAConvdtoi
; P8BE-LABEL: fromDiffMemConsAConvdtoi
; P8LE-LABEL: fromDiffMemConsAConvdtoi
; P9BE-DAG: lxv [[REG1:[vs0-9]+]], 0(r3)
; P9BE-DAG: lxv [[REG2:[vs0-9]+]], 16(r3)
; P9BE-DAG: xxmrgld [[REG3:[vs0-9]+]], [[REG1]], [[REG2]]
; P9BE-DAG: xxmrghd [[REG4:[vs0-9]+]], [[REG1]], [[REG2]]
; P9BE-DAG: xvcvdpsp [[REG5:[vs0-9]+]], [[REG3]]
; P9BE-DAG: xvcvdpsp [[REG6:[vs0-9]+]], [[REG4]]
; P9BE: vmrgew v2, [[REG6]], [[REG5]]
; P9BE: xvcvspsxws v2, v2
; P9LE-DAG: lxv [[REG1:[vs0-9]+]], 0(r3)
; P9LE-DAG: lxv [[REG2:[vs0-9]+]], 16(r3)
; P9LE-DAG: xxmrgld [[REG3:[vs0-9]+]], [[REG2]], [[REG1]]
; P9LE-DAG: xxmrghd [[REG4:[vs0-9]+]], [[REG2]], [[REG1]]
; P9LE-DAG: xvcvdpsp [[REG5:[vs0-9]+]], [[REG3]]
; P9LE-DAG: xvcvdpsp [[REG6:[vs0-9]+]], [[REG4]]
; P9LE: vmrgew v2, [[REG6]], [[REG5]]
; P9LE: xvcvspsxws v2, v2
; P8BE: lxvd2x [[REG1:[vs0-9]+]], 0, r3
; P8BE: lxvd2x [[REG2:[vs0-9]+]], r3, r4
; P8BE-DAG: xxmrgld [[REG3:[vs0-9]+]], [[REG1]], [[REG2]]
; P8BE-DAG: xxmrghd [[REG4:[vs0-9]+]], [[REG1]], [[REG2]]
; P8BE-DAG: xvcvdpsp [[REG5:[vs0-9]+]], [[REG3]]
; P8BE-DAG: xvcvdpsp [[REG6:[vs0-9]+]], [[REG4]]
; P8BE: vmrgew v2, [[REG6]], [[REG5]]
; P8BE: xvcvspsxws v2, v2
; P8LE: lxvd2x [[REG1:[vs0-9]+]], 0, r3
; P8LE: lxvd2x [[REG2:[vs0-9]+]], r3, r4
; P8LE-DAG: xxswapd [[REG3:[vs0-9]+]], [[REG1]]
; P8LE-DAG: xxswapd [[REG4:[vs0-9]+]], [[REG2]]
; P8LE-DAG: xxmrgld [[REG5:[vs0-9]+]], [[REG4]], [[REG3]]
; P8LE-DAG: xxmrghd [[REG6:[vs0-9]+]], [[REG4]], [[REG3]]
; P8LE-DAG: xvcvdpsp [[REG7:[vs0-9]+]], [[REG5]]
; P8LE-DAG: xvcvdpsp [[REG8:[vs0-9]+]], [[REG6]]
; P8LE: vmrgew v2, [[REG8]], [[REG7]]
; P8LE: xvcvspsxws v2, v2
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemConsDConvdtoi(double* nocapture readonly %ptr) {
entry:
  %arrayidx = getelementptr inbounds double, double* %ptr, i64 3
  %0 = load double, double* %arrayidx, align 8
  %conv = fptosi double %0 to i32
  %vecinit = insertelement <4 x i32> undef, i32 %conv, i32 0
  %arrayidx1 = getelementptr inbounds double, double* %ptr, i64 2
  %1 = load double, double* %arrayidx1, align 8
  %conv2 = fptosi double %1 to i32
  %vecinit3 = insertelement <4 x i32> %vecinit, i32 %conv2, i32 1
  %arrayidx4 = getelementptr inbounds double, double* %ptr, i64 1
  %2 = load double, double* %arrayidx4, align 8
  %conv5 = fptosi double %2 to i32
  %vecinit6 = insertelement <4 x i32> %vecinit3, i32 %conv5, i32 2
  %3 = load double, double* %ptr, align 8
  %conv8 = fptosi double %3 to i32
  %vecinit9 = insertelement <4 x i32> %vecinit6, i32 %conv8, i32 3
  ret <4 x i32> %vecinit9
; P9BE-LABEL: fromDiffMemConsDConvdtoi
; P9LE-LABEL: fromDiffMemConsDConvdtoi
; P8BE-LABEL: fromDiffMemConsDConvdtoi
; P8LE-LABEL: fromDiffMemConsDConvdtoi
; P9BE: lfd
; P9BE: lfd
; P9BE: lfd
; P9BE: lfd
; P9BE: xxmrghd
; P9BE: xxmrghd
; P9BE: xvcvdpsp
; P9BE: xvcvdpsp
; P9BE: vmrgew
; P9BE: xvcvspsxws v2
; P9LE: lfd
; P9LE: lfd
; P9LE: lfd
; P9LE: lfd
; P9LE: xxmrghd
; P9LE: xxmrghd
; P9LE: xvcvdpsp
; P9LE: xvcvdpsp
; P9LE: vmrgew
; P9LE: xvcvspsxws v2
; P8BE: lxsdx
; P8BE: lxsdx
; P8BE: lxsdx
; P8BE: lxsdx
; P8BE: xxmrghd
; P8BE: xxmrghd
; P8BE: xvcvdpsp
; P8BE: xvcvdpsp
; P8BE: vmrgew
; P8BE: xvcvspsxws v2
; P8LE: lxsdx
; P8LE: lxsdx
; P8LE: lxsdx
; P8LE: lxsdx
; P8LE: xxmrghd
; P8LE: xxmrghd
; P8LE: xvcvdpsp
; P8LE: xvcvdpsp
; P8LE: vmrgew
; P8LE: xvcvspsxws v2
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemVarAConvdtoi(double* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds double, double* %arr, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
  %conv = fptosi double %0 to i32
  %vecinit = insertelement <4 x i32> undef, i32 %conv, i32 0
  %add = add nsw i32 %elem, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds double, double* %arr, i64 %idxprom1
  %1 = load double, double* %arrayidx2, align 8
  %conv3 = fptosi double %1 to i32
  %vecinit4 = insertelement <4 x i32> %vecinit, i32 %conv3, i32 1
  %add5 = add nsw i32 %elem, 2
  %idxprom6 = sext i32 %add5 to i64
  %arrayidx7 = getelementptr inbounds double, double* %arr, i64 %idxprom6
  %2 = load double, double* %arrayidx7, align 8
  %conv8 = fptosi double %2 to i32
  %vecinit9 = insertelement <4 x i32> %vecinit4, i32 %conv8, i32 2
  %add10 = add nsw i32 %elem, 3
  %idxprom11 = sext i32 %add10 to i64
  %arrayidx12 = getelementptr inbounds double, double* %arr, i64 %idxprom11
  %3 = load double, double* %arrayidx12, align 8
  %conv13 = fptosi double %3 to i32
  %vecinit14 = insertelement <4 x i32> %vecinit9, i32 %conv13, i32 3
  ret <4 x i32> %vecinit14
; P9BE-LABEL: fromDiffMemVarAConvdtoi
; P9LE-LABEL: fromDiffMemVarAConvdtoi
; P8BE-LABEL: fromDiffMemVarAConvdtoi
; P8LE-LABEL: fromDiffMemVarAConvdtoi
; P9BE: lfdux
; P9BE: lfd
; P9BE: lfd
; P9BE: lfd
; P9BE: xxmrghd
; P9BE: xxmrghd
; P9BE: xvcvdpsp
; P9BE: xvcvdpsp
; P9BE: vmrgew
; P9BE: xvcvspsxws v2
; P9LE: lfdux
; P9LE: lfd
; P9LE: lfd
; P9LE: lfd
; P9LE: xxmrghd
; P9LE: xxmrghd
; P9LE: xvcvdpsp
; P9LE: xvcvdpsp
; P9LE: vmrgew
; P9LE: xvcvspsxws v2
; P8BE: lfdux
; P8BE: lxsdx
; P8BE: lxsdx
; P8BE: lxsdx
; P8BE: xxmrghd
; P8BE: xxmrghd
; P8BE: xvcvdpsp
; P8BE: xvcvdpsp
; P8BE: vmrgew
; P8BE: xvcvspsxws v2
; P8LE: lfdux
; P8LE: lxsdx
; P8LE: lxsdx
; P8LE: lxsdx
; P8LE: xxmrghd
; P8LE: xxmrghd
; P8LE: xvcvdpsp
; P8LE: xvcvdpsp
; P8LE: vmrgew
; P8LE: xvcvspsxws v2
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemVarDConvdtoi(double* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds double, double* %arr, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
  %conv = fptosi double %0 to i32
  %vecinit = insertelement <4 x i32> undef, i32 %conv, i32 0
  %sub = add nsw i32 %elem, -1
  %idxprom1 = sext i32 %sub to i64
  %arrayidx2 = getelementptr inbounds double, double* %arr, i64 %idxprom1
  %1 = load double, double* %arrayidx2, align 8
  %conv3 = fptosi double %1 to i32
  %vecinit4 = insertelement <4 x i32> %vecinit, i32 %conv3, i32 1
  %sub5 = add nsw i32 %elem, -2
  %idxprom6 = sext i32 %sub5 to i64
  %arrayidx7 = getelementptr inbounds double, double* %arr, i64 %idxprom6
  %2 = load double, double* %arrayidx7, align 8
  %conv8 = fptosi double %2 to i32
  %vecinit9 = insertelement <4 x i32> %vecinit4, i32 %conv8, i32 2
  %sub10 = add nsw i32 %elem, -3
  %idxprom11 = sext i32 %sub10 to i64
  %arrayidx12 = getelementptr inbounds double, double* %arr, i64 %idxprom11
  %3 = load double, double* %arrayidx12, align 8
  %conv13 = fptosi double %3 to i32
  %vecinit14 = insertelement <4 x i32> %vecinit9, i32 %conv13, i32 3
  ret <4 x i32> %vecinit14
; P9BE-LABEL: fromDiffMemVarDConvdtoi
; P9LE-LABEL: fromDiffMemVarDConvdtoi
; P8BE-LABEL: fromDiffMemVarDConvdtoi
; P8LE-LABEL: fromDiffMemVarDConvdtoi
; P9BE: lfdux
; P9BE: lfd
; P9BE: lfd
; P9BE: lfd
; P9BE: xxmrghd
; P9BE: xxmrghd
; P9BE: xvcvdpsp
; P9BE: xvcvdpsp
; P9BE: vmrgew
; P9BE: xvcvspsxws v2
; P9LE: lfdux
; P9LE: lfd
; P9LE: lfd
; P9LE: lfd
; P9LE: xxmrghd
; P9LE: xxmrghd
; P9LE: xvcvdpsp
; P9LE: xvcvdpsp
; P9LE: vmrgew
; P9LE: xvcvspsxws v2
; P8BE: lfdux
; P8BE: lxsdx
; P8BE: lxsdx
; P8BE: lxsdx
; P8BE: xxmrghd
; P8BE: xxmrghd
; P8BE: xvcvdpsp
; P8BE: xvcvdpsp
; P8BE: vmrgew
; P8BE: xvcvspsxws v2
; P8LE: lfdux
; P8LE: lxsdx
; P8LE: lxsdx
; P8LE: lxsdx
; P8LE: xxmrghd
; P8LE: xxmrghd
; P8LE: xvcvdpsp
; P8LE: xvcvdpsp
; P8LE: vmrgew
; P8LE: xvcvspsxws v2
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @spltRegValConvdtoi(double %val) {
entry:
  %conv = fptosi double %val to i32
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %conv, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; P9BE-LABEL: spltRegValConvdtoi
; P9LE-LABEL: spltRegValConvdtoi
; P8BE-LABEL: spltRegValConvdtoi
; P8LE-LABEL: spltRegValConvdtoi
; P9BE: xscvdpsxws
; P9BE: xxspltw
; P9BE: blr
; P9LE: xscvdpsxws
; P9LE: xxspltw
; P9LE: blr
; P8BE: xscvdpsxws
; P8BE: xxspltw
; P8BE: blr
; P8LE: xscvdpsxws
; P8LE: xxspltw
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @spltMemValConvdtoi(double* nocapture readonly %ptr) {
entry:
  %0 = load double, double* %ptr, align 8
  %conv = fptosi double %0 to i32
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %conv, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; P9BE-LABEL: spltMemValConvdtoi
; P9LE-LABEL: spltMemValConvdtoi
; P8BE-LABEL: spltMemValConvdtoi
; P8LE-LABEL: spltMemValConvdtoi
; P9BE: lfd
; P9BE: xscvdpsxws
; P9BE: xxspltw
; P9BE: blr
; P9LE: lfd
; P9LE: xscvdpsxws
; P9LE: xxspltw
; P9LE: blr
; P8BE: lfdx
; P8BE: xscvdpsxws
; P8BE: xxspltw
; P8BE: blr
; P8LE: lfdx
; P8LE: xscvdpsxws
; P8LE: xxspltw
; P8LE: blr
}
; Function Attrs: norecurse nounwind readnone
define <4 x i32> @allZeroui() {
entry:
  ret <4 x i32> zeroinitializer
; P9BE-LABEL: allZeroui
; P9LE-LABEL: allZeroui
; P8BE-LABEL: allZeroui
; P8LE-LABEL: allZeroui
; P9BE: xxlxor v2, v2, v2
; P9BE: blr
; P9LE: xxlxor v2, v2, v2
; P9LE: blr
; P8BE: xxlxor v2, v2, v2
; P8BE: blr
; P8LE: xxlxor v2, v2, v2
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @allOneui() {
entry:
  ret <4 x i32> <i32 -1, i32 -1, i32 -1, i32 -1>
; P9BE-LABEL: allOneui
; P9LE-LABEL: allOneui
; P8BE-LABEL: allOneui
; P8LE-LABEL: allOneui
; P9BE: xxspltib v2, 255
; P9BE: blr
; P9LE: xxspltib v2, 255
; P9LE: blr
; P8BE: vspltisb v2, -1
; P8BE: blr
; P8LE: vspltisb v2, -1
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @spltConst1ui() {
entry:
  ret <4 x i32> <i32 1, i32 1, i32 1, i32 1>
; P9BE-LABEL: spltConst1ui
; P9LE-LABEL: spltConst1ui
; P8BE-LABEL: spltConst1ui
; P8LE-LABEL: spltConst1ui
; P9BE: vspltisw v2, 1
; P9BE: blr
; P9LE: vspltisw v2, 1
; P9LE: blr
; P8BE: vspltisw v2, 1
; P8BE: blr
; P8LE: vspltisw v2, 1
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @spltConst16kui() {
entry:
  ret <4 x i32> <i32 32767, i32 32767, i32 32767, i32 32767>
; P9BE-LABEL: spltConst16kui
; P9LE-LABEL: spltConst16kui
; P8BE-LABEL: spltConst16kui
; P8LE-LABEL: spltConst16kui
; P9BE: vspltisw v2, -15
; P9BE: vsrw v2, v2, v2
; P9BE: blr
; P9LE: vspltisw v2, -15
; P9LE: vsrw v2, v2, v2
; P9LE: blr
; P8BE: vspltisw v2, -15
; P8BE: vsrw v2, v2, v2
; P8BE: blr
; P8LE: vspltisw v2, -15
; P8LE: vsrw v2, v2, v2
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @spltConst32kui() {
entry:
  ret <4 x i32> <i32 65535, i32 65535, i32 65535, i32 65535>
; P9BE-LABEL: spltConst32kui
; P9LE-LABEL: spltConst32kui
; P8BE-LABEL: spltConst32kui
; P8LE-LABEL: spltConst32kui
; P9BE: vspltisw v2, -16
; P9BE: vsrw v2, v2, v2
; P9BE: blr
; P9LE: vspltisw v2, -16
; P9LE: vsrw v2, v2, v2
; P9LE: blr
; P8BE: vspltisw v2, -16
; P8BE: vsrw v2, v2, v2
; P8BE: blr
; P8LE: vspltisw v2, -16
; P8LE: vsrw v2, v2, v2
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @fromRegsui(i32 zeroext %a, i32 zeroext %b, i32 zeroext %c, i32 zeroext %d) {
entry:
  %vecinit = insertelement <4 x i32> undef, i32 %a, i32 0
  %vecinit1 = insertelement <4 x i32> %vecinit, i32 %b, i32 1
  %vecinit2 = insertelement <4 x i32> %vecinit1, i32 %c, i32 2
  %vecinit3 = insertelement <4 x i32> %vecinit2, i32 %d, i32 3
  ret <4 x i32> %vecinit3
; P9BE-LABEL: fromRegsui
; P9LE-LABEL: fromRegsui
; P8BE-LABEL: fromRegsui
; P8LE-LABEL: fromRegsui
; P9BE-DAG: mtvsrdd [[REG1:v[0-9]+]], r3, r5
; P9BE-DAG: mtvsrdd [[REG2:v[0-9]+]], r4, r6
; P9BE: vmrgow v2, [[REG1]], [[REG2]]
; P9BE: blr
; P9LE-DAG: mtvsrdd [[REG1:v[0-9]+]], r5, r3
; P9LE-DAG: mtvsrdd [[REG2:v[0-9]+]], r6, r4
; P9LE: vmrgow v2, [[REG2]], [[REG1]]
; P9LE: blr
; P8BE-DAG: mtvsrwz {{[vf]}}[[REG1:[0-9]+]], r3
; P8BE-DAG: mtvsrwz {{[vf]}}[[REG2:[0-9]+]], r4
; P8BE-DAG: mtvsrwz {{[vf]}}[[REG3:[0-9]+]], r5
; P8BE-DAG: mtvsrwz {{[vf]}}[[REG4:[0-9]+]], r6
; P8BE-DAG: xxmrghd [[REG5:v[0-9]+]], {{[v][s]*}}[[REG1]], {{[v][s]*}}[[REG3]]
; P8BE-DAG: xxmrghd [[REG6:v[0-9]+]], {{[v][s]*}}[[REG2]], {{[v][s]*}}[[REG4]]
; P8BE: vmrgow v2, [[REG5]], [[REG6]]
; P8LE-DAG: mtvsrwz {{[vf]}}[[REG1:[0-9]+]], r3
; P8LE-DAG: mtvsrwz {{[vf]}}[[REG2:[0-9]+]], r4
; P8LE-DAG: mtvsrwz {{[vf]}}[[REG3:[0-9]+]], r5
; P8LE-DAG: mtvsrwz {{[vf]}}[[REG4:[0-9]+]], r6
; P8LE: xxmrghd [[REG5:v[0-9]+]], {{[v][s]*}}[[REG3]], {{[v][s]*}}[[REG1]]
; P8LE: xxmrghd [[REG6:v[0-9]+]], {{[v][s]*}}[[REG4]], {{[v][s]*}}[[REG2]]
; P8LE: vmrgow v2, [[REG6]], [[REG5]]
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @fromDiffConstsui() {
entry:
  ret <4 x i32> <i32 242, i32 -113, i32 889, i32 19>
; P9BE-LABEL: fromDiffConstsui
; P9LE-LABEL: fromDiffConstsui
; P8BE-LABEL: fromDiffConstsui
; P8LE-LABEL: fromDiffConstsui
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvw4x
; P8BE: blr
; P8LE: lvx
; P8LE-NOT: xxswapd
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemConsAui(i32* nocapture readonly %arr) {
entry:
  %0 = load i32, i32* %arr, align 4
  %vecinit = insertelement <4 x i32> undef, i32 %0, i32 0
  %arrayidx1 = getelementptr inbounds i32, i32* %arr, i64 1
  %1 = load i32, i32* %arrayidx1, align 4
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %1, i32 1
  %arrayidx3 = getelementptr inbounds i32, i32* %arr, i64 2
  %2 = load i32, i32* %arrayidx3, align 4
  %vecinit4 = insertelement <4 x i32> %vecinit2, i32 %2, i32 2
  %arrayidx5 = getelementptr inbounds i32, i32* %arr, i64 3
  %3 = load i32, i32* %arrayidx5, align 4
  %vecinit6 = insertelement <4 x i32> %vecinit4, i32 %3, i32 3
  ret <4 x i32> %vecinit6
; P9BE-LABEL: fromDiffMemConsAui
; P9LE-LABEL: fromDiffMemConsAui
; P8BE-LABEL: fromDiffMemConsAui
; P8LE-LABEL: fromDiffMemConsAui
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvw4x
; P8BE: blr
; P8LE: lxvd2x
; P8LE: xxswapd
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemConsDui(i32* nocapture readonly %arr) {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 3
  %0 = load i32, i32* %arrayidx, align 4
  %vecinit = insertelement <4 x i32> undef, i32 %0, i32 0
  %arrayidx1 = getelementptr inbounds i32, i32* %arr, i64 2
  %1 = load i32, i32* %arrayidx1, align 4
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %1, i32 1
  %arrayidx3 = getelementptr inbounds i32, i32* %arr, i64 1
  %2 = load i32, i32* %arrayidx3, align 4
  %vecinit4 = insertelement <4 x i32> %vecinit2, i32 %2, i32 2
  %3 = load i32, i32* %arr, align 4
  %vecinit6 = insertelement <4 x i32> %vecinit4, i32 %3, i32 3
  ret <4 x i32> %vecinit6
; P9BE-LABEL: fromDiffMemConsDui
; P9LE-LABEL: fromDiffMemConsDui
; P8BE-LABEL: fromDiffMemConsDui
; P8LE-LABEL: fromDiffMemConsDui
; P9BE: lxv
; P9BE: lxv
; P9BE: vperm
; P9BE: blr
; P9LE: lxv
; P9LE: lxv
; P9LE: vperm
; P9LE: blr
; P8BE: lxvw4x
; P8BE: lxvw4x
; P8BE: vperm
; P8BE: blr
; P8LE: lxvd2x
; P8LE-DAG: lvx
; P8LE-NOT: xxswapd
; P8LE: xxswapd
; P8LE: vperm
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemVarAui(i32* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %vecinit = insertelement <4 x i32> undef, i32 %0, i32 0
  %add = add nsw i32 %elem, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %arr, i64 %idxprom1
  %1 = load i32, i32* %arrayidx2, align 4
  %vecinit3 = insertelement <4 x i32> %vecinit, i32 %1, i32 1
  %add4 = add nsw i32 %elem, 2
  %idxprom5 = sext i32 %add4 to i64
  %arrayidx6 = getelementptr inbounds i32, i32* %arr, i64 %idxprom5
  %2 = load i32, i32* %arrayidx6, align 4
  %vecinit7 = insertelement <4 x i32> %vecinit3, i32 %2, i32 2
  %add8 = add nsw i32 %elem, 3
  %idxprom9 = sext i32 %add8 to i64
  %arrayidx10 = getelementptr inbounds i32, i32* %arr, i64 %idxprom9
  %3 = load i32, i32* %arrayidx10, align 4
  %vecinit11 = insertelement <4 x i32> %vecinit7, i32 %3, i32 3
  ret <4 x i32> %vecinit11
; P9BE-LABEL: fromDiffMemVarAui
; P9LE-LABEL: fromDiffMemVarAui
; P8BE-LABEL: fromDiffMemVarAui
; P8LE-LABEL: fromDiffMemVarAui
; P9BE: sldi r4, r4, 2
; P9BE: lxvx v2, r3, r4
; P9BE: blr
; P9LE: sldi r4, r4, 2
; P9LE: lxvx v2, r3, r4
; P9LE: blr
; P8BE: sldi r4, r4, 2
; P8BE: lxvw4x {{[vs0-9]+}}, r3, r4
; P8BE: blr
; P8LE: sldi r4, r4, 2
; P8LE: lxvd2x {{[vs0-9]+}}, r3, r4
; P8LE: xxswapd
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemVarDui(i32* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %vecinit = insertelement <4 x i32> undef, i32 %0, i32 0
  %sub = add nsw i32 %elem, -1
  %idxprom1 = sext i32 %sub to i64
  %arrayidx2 = getelementptr inbounds i32, i32* %arr, i64 %idxprom1
  %1 = load i32, i32* %arrayidx2, align 4
  %vecinit3 = insertelement <4 x i32> %vecinit, i32 %1, i32 1
  %sub4 = add nsw i32 %elem, -2
  %idxprom5 = sext i32 %sub4 to i64
  %arrayidx6 = getelementptr inbounds i32, i32* %arr, i64 %idxprom5
  %2 = load i32, i32* %arrayidx6, align 4
  %vecinit7 = insertelement <4 x i32> %vecinit3, i32 %2, i32 2
  %sub8 = add nsw i32 %elem, -3
  %idxprom9 = sext i32 %sub8 to i64
  %arrayidx10 = getelementptr inbounds i32, i32* %arr, i64 %idxprom9
  %3 = load i32, i32* %arrayidx10, align 4
  %vecinit11 = insertelement <4 x i32> %vecinit7, i32 %3, i32 3
  ret <4 x i32> %vecinit11
; P9BE-LABEL: fromDiffMemVarDui
; P9LE-LABEL: fromDiffMemVarDui
; P8BE-LABEL: fromDiffMemVarDui
; P8LE-LABEL: fromDiffMemVarDui
; P9BE-DAG: sldi {{r[0-9]+}}, r4, 2
; P9BE-DAG: addi r3, r3, -12
; P9BE-DAG: lxvx {{v[0-9]+}}, 0, r3
; P9BE-DAG: lxvx
; P9BE: vperm
; P9BE: blr
; P9LE-DAG: sldi {{r[0-9]+}}, r4, 2
; P9LE-DAG: addi r3, r3, -12
; P9LE-DAG: lxvx {{v[0-9]+}}, 0, r3
; P9LE-DAG: lxv
; P9LE: vperm
; P9LE: blr
; P8BE-DAG: sldi {{r[0-9]+}}, r4, 2
; P8BE-DAG: lxvw4x {{v[0-9]+}}, 0, r3
; P8BE-DAG: lxvw4x
; P8BE: vperm
; P8BE: blr
; P8LE-DAG: sldi {{r[0-9]+}}, r4, 2
; P8LE-DAG: lvx
; P8LE-DAG: lvx
; P8LE: vperm
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromRandMemConsui(i32* nocapture readonly %arr) {
entry:
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 4
  %0 = load i32, i32* %arrayidx, align 4
  %vecinit = insertelement <4 x i32> undef, i32 %0, i32 0
  %arrayidx1 = getelementptr inbounds i32, i32* %arr, i64 18
  %1 = load i32, i32* %arrayidx1, align 4
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %1, i32 1
  %arrayidx3 = getelementptr inbounds i32, i32* %arr, i64 2
  %2 = load i32, i32* %arrayidx3, align 4
  %vecinit4 = insertelement <4 x i32> %vecinit2, i32 %2, i32 2
  %arrayidx5 = getelementptr inbounds i32, i32* %arr, i64 88
  %3 = load i32, i32* %arrayidx5, align 4
  %vecinit6 = insertelement <4 x i32> %vecinit4, i32 %3, i32 3
  ret <4 x i32> %vecinit6
; P9BE-LABEL: fromRandMemConsui
; P9LE-LABEL: fromRandMemConsui
; P8BE-LABEL: fromRandMemConsui
; P8LE-LABEL: fromRandMemConsui
; P9BE: lwz
; P9BE: lwz
; P9BE: lwz
; P9BE: lwz
; P9BE: mtvsrdd
; P9BE: mtvsrdd
; P9BE: vmrgow
; P9LE: lwz
; P9LE: lwz
; P9LE: lwz
; P9LE: lwz
; P9LE: mtvsrdd
; P9LE: mtvsrdd
; P9LE: vmrgow
; P8BE: lwz
; P8BE: lwz
; P8BE: lwz
; P8BE: lwz
; P8BE: mtvsrwz
; P8BE: mtvsrwz
; P8BE: mtvsrwz
; P8BE: mtvsrwz
; P8BE: xxmrghd
; P8BE: xxmrghd
; P8BE: vmrgow
; P8LE: lwz
; P8LE: lwz
; P8LE: lwz
; P8LE: lwz
; P8LE: mtvsrwz
; P8LE: mtvsrwz
; P8LE: mtvsrwz
; P8LE: mtvsrwz
; P8LE: xxmrghd
; P8LE: xxmrghd
; P8LE: vmrgow
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromRandMemVarui(i32* nocapture readonly %arr, i32 signext %elem) {
entry:
  %add = add nsw i32 %elem, 4
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32, i32* %arr, i64 %idxprom
  %0 = load i32, i32* %arrayidx, align 4
  %vecinit = insertelement <4 x i32> undef, i32 %0, i32 0
  %add1 = add nsw i32 %elem, 1
  %idxprom2 = sext i32 %add1 to i64
  %arrayidx3 = getelementptr inbounds i32, i32* %arr, i64 %idxprom2
  %1 = load i32, i32* %arrayidx3, align 4
  %vecinit4 = insertelement <4 x i32> %vecinit, i32 %1, i32 1
  %add5 = add nsw i32 %elem, 2
  %idxprom6 = sext i32 %add5 to i64
  %arrayidx7 = getelementptr inbounds i32, i32* %arr, i64 %idxprom6
  %2 = load i32, i32* %arrayidx7, align 4
  %vecinit8 = insertelement <4 x i32> %vecinit4, i32 %2, i32 2
  %add9 = add nsw i32 %elem, 8
  %idxprom10 = sext i32 %add9 to i64
  %arrayidx11 = getelementptr inbounds i32, i32* %arr, i64 %idxprom10
  %3 = load i32, i32* %arrayidx11, align 4
  %vecinit12 = insertelement <4 x i32> %vecinit8, i32 %3, i32 3
  ret <4 x i32> %vecinit12
; P9BE-LABEL: fromRandMemVarui
; P9LE-LABEL: fromRandMemVarui
; P8BE-LABEL: fromRandMemVarui
; P8LE-LABEL: fromRandMemVarui
; P9BE: sldi r4, r4, 2
; P9BE: lwz
; P9BE: lwz
; P9BE: lwz
; P9BE: lwz
; P9BE: mtvsrdd
; P9BE: mtvsrdd
; P9BE: vmrgow
; P9LE: sldi r4, r4, 2
; P9LE: lwz
; P9LE: lwz
; P9LE: lwz
; P9LE: lwz
; P9LE: mtvsrdd
; P9LE: mtvsrdd
; P9LE: vmrgow
; P8BE: sldi r4, r4, 2
; P8BE: lwz
; P8BE: lwz
; P8BE: lwz
; P8BE: lwz
; P8BE: mtvsrwz
; P8BE: mtvsrwz
; P8BE: mtvsrwz
; P8BE: mtvsrwz
; P8BE: xxmrghd
; P8BE: xxmrghd
; P8BE: vmrgow
; P8LE: sldi r4, r4, 2
; P8LE: lwz
; P8LE: lwz
; P8LE: lwz
; P8LE: lwz
; P8LE: mtvsrwz
; P8LE: mtvsrwz
; P8LE: mtvsrwz
; P8LE: mtvsrwz
; P8LE: xxmrghd
; P8LE: xxmrghd
; P8LE: vmrgow
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @spltRegValui(i32 zeroext %val) {
entry:
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %val, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; P9BE-LABEL: spltRegValui
; P9LE-LABEL: spltRegValui
; P8BE-LABEL: spltRegValui
; P8LE-LABEL: spltRegValui
; P9BE: mtvsrws v2, r3
; P9BE: blr
; P9LE: mtvsrws v2, r3
; P9LE: blr
; P8BE: mtvsrwz {{[vsf0-9]+}}, r3
; P8BE: xxspltw v2, {{[vsf0-9]+}}, 1
; P8BE: blr
; P8LE: mtvsrwz {{[vsf0-9]+}}, r3
; P8LE: xxspltw v2, {{[vsf0-9]+}}, 1
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @spltMemValui(i32* nocapture readonly %ptr) {
entry:
  %0 = load i32, i32* %ptr, align 4
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %0, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; P9BE-LABEL: spltMemValui
; P9LE-LABEL: spltMemValui
; P8BE-LABEL: spltMemValui
; P8LE-LABEL: spltMemValui
; P9BE: lxvwsx v2, 0, r3
; P9BE: blr
; P9LE: lxvwsx v2, 0, r3
; P9LE: blr
; P8BE: lxsiwax {{[vsf0-9]+}}, 0, r3
; P8BE: xxspltw v2, {{[vsf0-9]+}}, 1
; P8BE: blr
; P8LE: lxsiwax {{[vsf0-9]+}}, 0, r3
; P8LE: xxspltw v2, {{[vsf0-9]+}}, 1
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @spltCnstConvftoui() {
entry:
  ret <4 x i32> <i32 4, i32 4, i32 4, i32 4>
; P9BE-LABEL: spltCnstConvftoui
; P9LE-LABEL: spltCnstConvftoui
; P8BE-LABEL: spltCnstConvftoui
; P8LE-LABEL: spltCnstConvftoui
; P9BE: vspltisw v2, 4
; P9BE: blr
; P9LE: vspltisw v2, 4
; P9LE: blr
; P8BE: vspltisw v2, 4
; P8BE: blr
; P8LE: vspltisw v2, 4
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @fromRegsConvftoui(float %a, float %b, float %c, float %d) {
entry:
  %conv = fptoui float %a to i32
  %vecinit = insertelement <4 x i32> undef, i32 %conv, i32 0
  %conv1 = fptoui float %b to i32
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %conv1, i32 1
  %conv3 = fptoui float %c to i32
  %vecinit4 = insertelement <4 x i32> %vecinit2, i32 %conv3, i32 2
  %conv5 = fptoui float %d to i32
  %vecinit6 = insertelement <4 x i32> %vecinit4, i32 %conv5, i32 3
  ret <4 x i32> %vecinit6
; P9BE-LABEL: fromRegsConvftoui
; P9LE-LABEL: fromRegsConvftoui
; P8BE-LABEL: fromRegsConvftoui
; P8LE-LABEL: fromRegsConvftoui
; P9BE-DAG: xxmrghd {{[vs]+}}[[REG1:[0-9]+]], vs1, vs3
; P9BE-DAG: xxmrghd {{[vs]+}}[[REG2:[0-9]+]], vs2, vs4
; P9BE-DAG: xvcvdpsp [[REG3:v[0-9]+]], {{[vs]+}}[[REG1]]
; P9BE-DAG: xvcvdpsp [[REG4:v[0-9]+]], {{[vs]+}}[[REG2]]
; P9BE: vmrgew v2, [[REG3]], [[REG4]]
; P9BE: xvcvspuxws v2, v2
; P9LE-DAG: xxmrghd {{[vs]+}}[[REG1:[0-9]+]], vs3, vs1
; P9LE-DAG: xxmrghd {{[vs]+}}[[REG2:[0-9]+]], vs4, vs2
; P9LE-DAG: xvcvdpsp [[REG3:v[0-9]+]], {{[vs]+}}[[REG1]]
; P9LE-DAG: xvcvdpsp [[REG4:v[0-9]+]], {{[vs]+}}[[REG2]]
; P9LE: vmrgew v2, [[REG4]], [[REG3]]
; P9LE: xvcvspuxws v2, v2
; P8BE-DAG: xxmrghd {{[vs]+}}[[REG1:[0-9]+]], vs1, vs3
; P8BE-DAG: xxmrghd {{[vs]+}}[[REG2:[0-9]+]], vs2, vs4
; P8BE-DAG: xvcvdpsp [[REG3:v[0-9]+]], {{[vs]+}}[[REG1]]
; P8BE-DAG: xvcvdpsp [[REG4:v[0-9]+]], {{[vs]+}}[[REG2]]
; P8BE: vmrgew v2, [[REG3]], [[REG4]]
; P8BE: xvcvspuxws v2, v2
; P8LE-DAG: xxmrghd {{[vs]+}}[[REG1:[0-9]+]], vs3, vs1
; P8LE-DAG: xxmrghd {{[vs]+}}[[REG2:[0-9]+]], vs4, vs2
; P8LE-DAG: xvcvdpsp [[REG3:v[0-9]+]], {{[vs]+}}[[REG1]]
; P8LE-DAG: xvcvdpsp [[REG4:v[0-9]+]], {{[vs]+}}[[REG2]]
; P8LE: vmrgew v2, [[REG4]], [[REG3]]
; P8LE: xvcvspuxws v2, v2
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @fromDiffConstsConvftoui() {
entry:
  ret <4 x i32> <i32 24, i32 234, i32 988, i32 422>
; P9BE-LABEL: fromDiffConstsConvftoui
; P9LE-LABEL: fromDiffConstsConvftoui
; P8BE-LABEL: fromDiffConstsConvftoui
; P8LE-LABEL: fromDiffConstsConvftoui
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvw4x
; P8BE: blr
; P8LE: lvx
; P8LE-NOT: xxswapd
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemConsAConvftoui(float* nocapture readonly %ptr) {
entry:
  %0 = bitcast float* %ptr to <4 x float>*
  %1 = load <4 x float>, <4 x float>* %0, align 4
  %2 = fptoui <4 x float> %1 to <4 x i32>
  ret <4 x i32> %2
; P9BE-LABEL: fromDiffMemConsAConvftoui
; P9LE-LABEL: fromDiffMemConsAConvftoui
; P8BE-LABEL: fromDiffMemConsAConvftoui
; P8LE-LABEL: fromDiffMemConsAConvftoui
; P9BE: lxv [[REG1:[vs0-9]+]], 0(r3)
; P9BE: xvcvspuxws v2, [[REG1]]
; P9BE: blr
; P9LE: lxv [[REG1:[vs0-9]+]], 0(r3)
; P9LE: xvcvspuxws v2, [[REG1]]
; P9LE: blr
; P8BE: lxvw4x [[REG1:[vs0-9]+]], 0, r3
; P8BE: xvcvspuxws v2, [[REG1]]
; P8BE: blr
; P8LE: lxvd2x [[REG1:[vs0-9]+]], 0, r3
; P8LE: xxswapd v2, [[REG1]]
; P8LE: xvcvspuxws v2, v2
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemConsDConvftoui(float* nocapture readonly %ptr) {
entry:
  %arrayidx = getelementptr inbounds float, float* %ptr, i64 3
  %0 = load float, float* %arrayidx, align 4
  %conv = fptoui float %0 to i32
  %vecinit = insertelement <4 x i32> undef, i32 %conv, i32 0
  %arrayidx1 = getelementptr inbounds float, float* %ptr, i64 2
  %1 = load float, float* %arrayidx1, align 4
  %conv2 = fptoui float %1 to i32
  %vecinit3 = insertelement <4 x i32> %vecinit, i32 %conv2, i32 1
  %arrayidx4 = getelementptr inbounds float, float* %ptr, i64 1
  %2 = load float, float* %arrayidx4, align 4
  %conv5 = fptoui float %2 to i32
  %vecinit6 = insertelement <4 x i32> %vecinit3, i32 %conv5, i32 2
  %3 = load float, float* %ptr, align 4
  %conv8 = fptoui float %3 to i32
  %vecinit9 = insertelement <4 x i32> %vecinit6, i32 %conv8, i32 3
  ret <4 x i32> %vecinit9
; P9BE-LABEL: fromDiffMemConsDConvftoui
; P9LE-LABEL: fromDiffMemConsDConvftoui
; P8BE-LABEL: fromDiffMemConsDConvftoui
; P8LE-LABEL: fromDiffMemConsDConvftoui
; P9BE: lxv
; P9BE: lxv
; P9BE: vperm
; P9BE: xvcvspuxws
; P9BE: blr
; P9LE: lxv
; P9LE: lxv
; P9LE: vperm
; P9LE: xvcvspuxws
; P9LE: blr
; P8BE: lxvw4x
; P8BE: lxvw4x
; P8BE: vperm
; P8BE: xvcvspuxws
; P8BE: blr
; P8LE-DAG: lxvd2x
; P8LE-DAG: lvx
; P8LE: xxswapd
; P8LE: vperm
; P8LE: xvcvspuxws
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemVarAConvftoui(float* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds float, float* %arr, i64 %idxprom
  %0 = load float, float* %arrayidx, align 4
  %conv = fptoui float %0 to i32
  %vecinit = insertelement <4 x i32> undef, i32 %conv, i32 0
  %add = add nsw i32 %elem, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds float, float* %arr, i64 %idxprom1
  %1 = load float, float* %arrayidx2, align 4
  %conv3 = fptoui float %1 to i32
  %vecinit4 = insertelement <4 x i32> %vecinit, i32 %conv3, i32 1
  %add5 = add nsw i32 %elem, 2
  %idxprom6 = sext i32 %add5 to i64
  %arrayidx7 = getelementptr inbounds float, float* %arr, i64 %idxprom6
  %2 = load float, float* %arrayidx7, align 4
  %conv8 = fptoui float %2 to i32
  %vecinit9 = insertelement <4 x i32> %vecinit4, i32 %conv8, i32 2
  %add10 = add nsw i32 %elem, 3
  %idxprom11 = sext i32 %add10 to i64
  %arrayidx12 = getelementptr inbounds float, float* %arr, i64 %idxprom11
  %3 = load float, float* %arrayidx12, align 4
  %conv13 = fptoui float %3 to i32
  %vecinit14 = insertelement <4 x i32> %vecinit9, i32 %conv13, i32 3
  ret <4 x i32> %vecinit14
; P9BE-LABEL: fromDiffMemVarAConvftoui
; P9LE-LABEL: fromDiffMemVarAConvftoui
; P8BE-LABEL: fromDiffMemVarAConvftoui
; P8LE-LABEL: fromDiffMemVarAConvftoui
; FIXME: implement finding consecutive loads with pre-inc
; P9BE: lfsux
; P9LE: lfsux
; P8BE: lfsux
; P8LE: lfsux
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemVarDConvftoui(float* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds float, float* %arr, i64 %idxprom
  %0 = load float, float* %arrayidx, align 4
  %conv = fptoui float %0 to i32
  %vecinit = insertelement <4 x i32> undef, i32 %conv, i32 0
  %sub = add nsw i32 %elem, -1
  %idxprom1 = sext i32 %sub to i64
  %arrayidx2 = getelementptr inbounds float, float* %arr, i64 %idxprom1
  %1 = load float, float* %arrayidx2, align 4
  %conv3 = fptoui float %1 to i32
  %vecinit4 = insertelement <4 x i32> %vecinit, i32 %conv3, i32 1
  %sub5 = add nsw i32 %elem, -2
  %idxprom6 = sext i32 %sub5 to i64
  %arrayidx7 = getelementptr inbounds float, float* %arr, i64 %idxprom6
  %2 = load float, float* %arrayidx7, align 4
  %conv8 = fptoui float %2 to i32
  %vecinit9 = insertelement <4 x i32> %vecinit4, i32 %conv8, i32 2
  %sub10 = add nsw i32 %elem, -3
  %idxprom11 = sext i32 %sub10 to i64
  %arrayidx12 = getelementptr inbounds float, float* %arr, i64 %idxprom11
  %3 = load float, float* %arrayidx12, align 4
  %conv13 = fptoui float %3 to i32
  %vecinit14 = insertelement <4 x i32> %vecinit9, i32 %conv13, i32 3
  ret <4 x i32> %vecinit14
; P9BE-LABEL: fromDiffMemVarDConvftoui
; P9LE-LABEL: fromDiffMemVarDConvftoui
; P8BE-LABEL: fromDiffMemVarDConvftoui
; P8LE-LABEL: fromDiffMemVarDConvftoui
; FIXME: implement finding consecutive loads with pre-inc
; P9BE: lfsux
; P9LE: lfsux
; P8BE: lfsux
; P8LE: lfsux
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @spltRegValConvftoui(float %val) {
entry:
  %conv = fptoui float %val to i32
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %conv, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; P9BE-LABEL: spltRegValConvftoui
; P9LE-LABEL: spltRegValConvftoui
; P8BE-LABEL: spltRegValConvftoui
; P8LE-LABEL: spltRegValConvftoui
; P9BE: xscvdpuxws f[[REG1:[0-9]+]], f1
; P9BE: xxspltw v2, vs[[REG1]], 1
; P9BE: blr
; P9LE: xscvdpuxws f[[REG1:[0-9]+]], f1
; P9LE: xxspltw v2, vs[[REG1]], 1
; P9LE: blr
; P8BE: xscvdpuxws f[[REG1:[0-9]+]], f1
; P8BE: xxspltw v2, vs[[REG1]], 1
; P8BE: blr
; P8LE: xscvdpuxws f[[REG1:[0-9]+]], f1
; P8LE: xxspltw v2, vs[[REG1]], 1
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @spltMemValConvftoui(float* nocapture readonly %ptr) {
entry:
  %0 = load float, float* %ptr, align 4
  %conv = fptoui float %0 to i32
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %conv, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; P9BE-LABEL: spltMemValConvftoui
; P9LE-LABEL: spltMemValConvftoui
; P8BE-LABEL: spltMemValConvftoui
; P8LE-LABEL: spltMemValConvftoui
; P9BE: lxvwsx [[REG1:[vs0-9]+]], 0, r3
; P9BE: xvcvspuxws v2, [[REG1]]
; P9LE: [[REG1:[vs0-9]+]], 0, r3
; P9LE: xvcvspuxws v2, [[REG1]]
; P8BE: lfsx [[REG1:f[0-9]+]], 0, r3
; P8BE: xscvdpuxws f[[REG2:[0-9]+]], [[REG1]]
; P8BE: xxspltw v2, vs[[REG2]], 1
; P8LE: lfsx [[REG1:f[0-9]+]], 0, r3
; P8LE: xscvdpuxws f[[REG2:[vs0-9]+]], [[REG1]]
; P8LE: xxspltw v2, vs[[REG2]], 1
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @spltCnstConvdtoui() {
entry:
  ret <4 x i32> <i32 4, i32 4, i32 4, i32 4>
; P9BE-LABEL: spltCnstConvdtoui
; P9LE-LABEL: spltCnstConvdtoui
; P8BE-LABEL: spltCnstConvdtoui
; P8LE-LABEL: spltCnstConvdtoui
; P9BE: vspltisw v2, 4
; P9BE: blr
; P9LE: vspltisw v2, 4
; P9LE: blr
; P8BE: vspltisw v2, 4
; P8BE: blr
; P8LE: vspltisw v2, 4
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @fromRegsConvdtoui(double %a, double %b, double %c, double %d) {
entry:
  %conv = fptoui double %a to i32
  %vecinit = insertelement <4 x i32> undef, i32 %conv, i32 0
  %conv1 = fptoui double %b to i32
  %vecinit2 = insertelement <4 x i32> %vecinit, i32 %conv1, i32 1
  %conv3 = fptoui double %c to i32
  %vecinit4 = insertelement <4 x i32> %vecinit2, i32 %conv3, i32 2
  %conv5 = fptoui double %d to i32
  %vecinit6 = insertelement <4 x i32> %vecinit4, i32 %conv5, i32 3
  ret <4 x i32> %vecinit6
; P9BE-LABEL: fromRegsConvdtoui
; P9LE-LABEL: fromRegsConvdtoui
; P8BE-LABEL: fromRegsConvdtoui
; P8LE-LABEL: fromRegsConvdtoui
; P9BE-DAG: xxmrghd {{[vs]+}}[[REG1:[0-9]+]], vs1, vs3
; P9BE-DAG: xxmrghd {{[vs]+}}[[REG2:[0-9]+]], vs2, vs4
; P9BE-DAG: xvcvdpsp [[REG3:v[0-9]+]], {{[vs]+}}[[REG1]]
; P9BE-DAG: xvcvdpsp [[REG4:v[0-9]+]], {{[vs]+}}[[REG2]]
; P9BE: vmrgew v2, [[REG3]], [[REG4]]
; P9BE: xvcvspuxws v2, v2
; P9LE-DAG: xxmrghd {{[vs]+}}[[REG1:[0-9]+]], vs3, vs1
; P9LE-DAG: xxmrghd {{[vs]+}}[[REG2:[0-9]+]], vs4, vs2
; P9LE-DAG: xvcvdpsp [[REG3:v[0-9]+]], {{[vs]+}}[[REG1]]
; P9LE-DAG: xvcvdpsp [[REG4:v[0-9]+]], {{[vs]+}}[[REG2]]
; P9LE: vmrgew v2, [[REG4]], [[REG3]]
; P9LE: xvcvspuxws v2, v2
; P8BE-DAG: xxmrghd {{[vs]+}}[[REG1:[0-9]+]], vs1, vs3
; P8BE-DAG: xxmrghd {{[vs]+}}[[REG2:[0-9]+]], vs2, vs4
; P8BE-DAG: xvcvdpsp [[REG3:v[0-9]+]], {{[vs]+}}[[REG1]]
; P8BE-DAG: xvcvdpsp [[REG4:v[0-9]+]], {{[vs]+}}[[REG2]]
; P8BE: vmrgew v2, [[REG3]], [[REG4]]
; P8BE: xvcvspuxws v2, v2
; P8LE-DAG: xxmrghd {{[vs]+}}[[REG1:[0-9]+]], vs3, vs1
; P8LE-DAG: xxmrghd {{[vs]+}}[[REG2:[0-9]+]], vs4, vs2
; P8LE-DAG: xvcvdpsp [[REG3:v[0-9]+]], {{[vs]+}}[[REG1]]
; P8LE-DAG: xvcvdpsp [[REG4:v[0-9]+]], {{[vs]+}}[[REG2]]
; P8LE: vmrgew v2, [[REG4]], [[REG3]]
; P8LE: xvcvspuxws v2, v2
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @fromDiffConstsConvdtoui() {
entry:
  ret <4 x i32> <i32 24, i32 234, i32 988, i32 422>
; P9BE-LABEL: fromDiffConstsConvdtoui
; P9LE-LABEL: fromDiffConstsConvdtoui
; P8BE-LABEL: fromDiffConstsConvdtoui
; P8LE-LABEL: fromDiffConstsConvdtoui
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvw4x
; P8BE: blr
; P8LE: lvx
; P8LE-NOT: xxswapd
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemConsAConvdtoui(double* nocapture readonly %ptr) {
entry:
  %0 = bitcast double* %ptr to <2 x double>*
  %1 = load <2 x double>, <2 x double>* %0, align 8
  %2 = fptoui <2 x double> %1 to <2 x i32>
  %arrayidx4 = getelementptr inbounds double, double* %ptr, i64 2
  %3 = bitcast double* %arrayidx4 to <2 x double>*
  %4 = load <2 x double>, <2 x double>* %3, align 8
  %5 = fptoui <2 x double> %4 to <2 x i32>
  %vecinit9 = shufflevector <2 x i32> %2, <2 x i32> %5, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x i32> %vecinit9
; P9BE-LABEL: fromDiffMemConsAConvdtoui
; P9LE-LABEL: fromDiffMemConsAConvdtoui
; P8BE-LABEL: fromDiffMemConsAConvdtoui
; P8LE-LABEL: fromDiffMemConsAConvdtoui
; P9BE-DAG: lxv [[REG1:[vs0-9]+]], 0(r3)
; P9BE-DAG: lxv [[REG2:[vs0-9]+]], 16(r3)
; P9BE-DAG: xxmrgld [[REG3:[vs0-9]+]], [[REG1]], [[REG2]]
; P9BE-DAG: xxmrghd [[REG4:[vs0-9]+]], [[REG1]], [[REG2]]
; P9BE-DAG: xvcvdpsp [[REG5:[vs0-9]+]], [[REG3]]
; P9BE-DAG: xvcvdpsp [[REG6:[vs0-9]+]], [[REG4]]
; P9BE: vmrgew v2, [[REG6]], [[REG5]]
; P9BE: xvcvspuxws v2, v2
; P9LE-DAG: lxv [[REG1:[vs0-9]+]], 0(r3)
; P9LE-DAG: lxv [[REG2:[vs0-9]+]], 16(r3)
; P9LE-DAG: xxmrgld [[REG3:[vs0-9]+]], [[REG2]], [[REG1]]
; P9LE-DAG: xxmrghd [[REG4:[vs0-9]+]], [[REG2]], [[REG1]]
; P9LE-DAG: xvcvdpsp [[REG5:[vs0-9]+]], [[REG3]]
; P9LE-DAG: xvcvdpsp [[REG6:[vs0-9]+]], [[REG4]]
; P9LE: vmrgew v2, [[REG6]], [[REG5]]
; P9LE: xvcvspuxws v2, v2
; P8BE: lxvd2x [[REG1:[vs0-9]+]], 0, r3
; P8BE: lxvd2x [[REG2:[vs0-9]+]], r3, r4
; P8BE-DAG: xxmrgld [[REG3:[vs0-9]+]], [[REG1]], [[REG2]]
; P8BE-DAG: xxmrghd [[REG4:[vs0-9]+]], [[REG1]], [[REG2]]
; P8BE-DAG: xvcvdpsp [[REG5:[vs0-9]+]], [[REG3]]
; P8BE-DAG: xvcvdpsp [[REG6:[vs0-9]+]], [[REG4]]
; P8BE: vmrgew v2, [[REG6]], [[REG5]]
; P8BE: xvcvspuxws v2, v2
; P8LE: lxvd2x [[REG1:[vs0-9]+]], 0, r3
; P8LE: lxvd2x [[REG2:[vs0-9]+]], r3, r4
; P8LE-DAG: xxswapd [[REG3:[vs0-9]+]], [[REG1]]
; P8LE-DAG: xxswapd [[REG4:[vs0-9]+]], [[REG2]]
; P8LE-DAG: xxmrgld [[REG5:[vs0-9]+]], [[REG4]], [[REG3]]
; P8LE-DAG: xxmrghd [[REG6:[vs0-9]+]], [[REG4]], [[REG3]]
; P8LE-DAG: xvcvdpsp [[REG7:[vs0-9]+]], [[REG5]]
; P8LE-DAG: xvcvdpsp [[REG8:[vs0-9]+]], [[REG6]]
; P8LE: vmrgew v2, [[REG8]], [[REG7]]
; P8LE: xvcvspuxws v2, v2
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemConsDConvdtoui(double* nocapture readonly %ptr) {
entry:
  %arrayidx = getelementptr inbounds double, double* %ptr, i64 3
  %0 = load double, double* %arrayidx, align 8
  %conv = fptoui double %0 to i32
  %vecinit = insertelement <4 x i32> undef, i32 %conv, i32 0
  %arrayidx1 = getelementptr inbounds double, double* %ptr, i64 2
  %1 = load double, double* %arrayidx1, align 8
  %conv2 = fptoui double %1 to i32
  %vecinit3 = insertelement <4 x i32> %vecinit, i32 %conv2, i32 1
  %arrayidx4 = getelementptr inbounds double, double* %ptr, i64 1
  %2 = load double, double* %arrayidx4, align 8
  %conv5 = fptoui double %2 to i32
  %vecinit6 = insertelement <4 x i32> %vecinit3, i32 %conv5, i32 2
  %3 = load double, double* %ptr, align 8
  %conv8 = fptoui double %3 to i32
  %vecinit9 = insertelement <4 x i32> %vecinit6, i32 %conv8, i32 3
  ret <4 x i32> %vecinit9
; P9BE-LABEL: fromDiffMemConsDConvdtoui
; P9LE-LABEL: fromDiffMemConsDConvdtoui
; P8BE-LABEL: fromDiffMemConsDConvdtoui
; P8LE-LABEL: fromDiffMemConsDConvdtoui
; P9BE: lfd
; P9BE: lfd
; P9BE: lfd
; P9BE: lfd
; P9BE: xxmrghd
; P9BE: xxmrghd
; P9BE: xvcvdpsp
; P9BE: xvcvdpsp
; P9BE: vmrgew
; P9BE: xvcvspuxws v2
; P9LE: lfd
; P9LE: lfd
; P9LE: lfd
; P9LE: lfd
; P9LE: xxmrghd
; P9LE: xxmrghd
; P9LE: xvcvdpsp
; P9LE: xvcvdpsp
; P9LE: vmrgew
; P9LE: xvcvspuxws v2
; P8BE: lxsdx
; P8BE: lxsdx
; P8BE: lxsdx
; P8BE: lxsdx
; P8BE: xxmrghd
; P8BE: xxmrghd
; P8BE: xvcvdpsp
; P8BE: xvcvdpsp
; P8BE: vmrgew
; P8BE: xvcvspuxws v2
; P8LE: lxsdx
; P8LE: lxsdx
; P8LE: lxsdx
; P8LE: lxsdx
; P8LE: xxmrghd
; P8LE: xxmrghd
; P8LE: xvcvdpsp
; P8LE: xvcvdpsp
; P8LE: vmrgew
; P8LE: xvcvspuxws v2
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemVarAConvdtoui(double* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds double, double* %arr, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
  %conv = fptoui double %0 to i32
  %vecinit = insertelement <4 x i32> undef, i32 %conv, i32 0
  %add = add nsw i32 %elem, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds double, double* %arr, i64 %idxprom1
  %1 = load double, double* %arrayidx2, align 8
  %conv3 = fptoui double %1 to i32
  %vecinit4 = insertelement <4 x i32> %vecinit, i32 %conv3, i32 1
  %add5 = add nsw i32 %elem, 2
  %idxprom6 = sext i32 %add5 to i64
  %arrayidx7 = getelementptr inbounds double, double* %arr, i64 %idxprom6
  %2 = load double, double* %arrayidx7, align 8
  %conv8 = fptoui double %2 to i32
  %vecinit9 = insertelement <4 x i32> %vecinit4, i32 %conv8, i32 2
  %add10 = add nsw i32 %elem, 3
  %idxprom11 = sext i32 %add10 to i64
  %arrayidx12 = getelementptr inbounds double, double* %arr, i64 %idxprom11
  %3 = load double, double* %arrayidx12, align 8
  %conv13 = fptoui double %3 to i32
  %vecinit14 = insertelement <4 x i32> %vecinit9, i32 %conv13, i32 3
  ret <4 x i32> %vecinit14
; P9BE-LABEL: fromDiffMemVarAConvdtoui
; P9LE-LABEL: fromDiffMemVarAConvdtoui
; P8BE-LABEL: fromDiffMemVarAConvdtoui
; P8LE-LABEL: fromDiffMemVarAConvdtoui
; P9BE: lfdux
; P9BE: lfd
; P9BE: lfd
; P9BE: lfd
; P9BE: xxmrghd
; P9BE: xxmrghd
; P9BE: xvcvdpsp
; P9BE: xvcvdpsp
; P9BE: vmrgew
; P9BE: xvcvspuxws v2
; P9LE: lfdux
; P9LE: lfd
; P9LE: lfd
; P9LE: lfd
; P9LE: xxmrghd
; P9LE: xxmrghd
; P9LE: xvcvdpsp
; P9LE: xvcvdpsp
; P9LE: vmrgew
; P9LE: xvcvspuxws v2
; P8BE: lfdux
; P8BE: lxsdx
; P8BE: lxsdx
; P8BE: lxsdx
; P8BE: xxmrghd
; P8BE: xxmrghd
; P8BE: xvcvdpsp
; P8BE: xvcvdpsp
; P8BE: vmrgew
; P8BE: xvcvspuxws v2
; P8LE: lfdux
; P8LE: lxsdx
; P8LE: lxsdx
; P8LE: lxsdx
; P8LE: xxmrghd
; P8LE: xxmrghd
; P8LE: xvcvdpsp
; P8LE: xvcvdpsp
; P8LE: vmrgew
; P8LE: xvcvspuxws v2
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @fromDiffMemVarDConvdtoui(double* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds double, double* %arr, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
  %conv = fptoui double %0 to i32
  %vecinit = insertelement <4 x i32> undef, i32 %conv, i32 0
  %sub = add nsw i32 %elem, -1
  %idxprom1 = sext i32 %sub to i64
  %arrayidx2 = getelementptr inbounds double, double* %arr, i64 %idxprom1
  %1 = load double, double* %arrayidx2, align 8
  %conv3 = fptoui double %1 to i32
  %vecinit4 = insertelement <4 x i32> %vecinit, i32 %conv3, i32 1
  %sub5 = add nsw i32 %elem, -2
  %idxprom6 = sext i32 %sub5 to i64
  %arrayidx7 = getelementptr inbounds double, double* %arr, i64 %idxprom6
  %2 = load double, double* %arrayidx7, align 8
  %conv8 = fptoui double %2 to i32
  %vecinit9 = insertelement <4 x i32> %vecinit4, i32 %conv8, i32 2
  %sub10 = add nsw i32 %elem, -3
  %idxprom11 = sext i32 %sub10 to i64
  %arrayidx12 = getelementptr inbounds double, double* %arr, i64 %idxprom11
  %3 = load double, double* %arrayidx12, align 8
  %conv13 = fptoui double %3 to i32
  %vecinit14 = insertelement <4 x i32> %vecinit9, i32 %conv13, i32 3
  ret <4 x i32> %vecinit14
; P9BE-LABEL: fromDiffMemVarDConvdtoui
; P9LE-LABEL: fromDiffMemVarDConvdtoui
; P8BE-LABEL: fromDiffMemVarDConvdtoui
; P8LE-LABEL: fromDiffMemVarDConvdtoui
; P9BE: lfdux
; P9BE: lfd
; P9BE: lfd
; P9BE: lfd
; P9BE: xxmrghd
; P9BE: xxmrghd
; P9BE: xvcvdpsp
; P9BE: xvcvdpsp
; P9BE: vmrgew
; P9BE: xvcvspuxws v2
; P9LE: lfdux
; P9LE: lfd
; P9LE: lfd
; P9LE: lfd
; P9LE: xxmrghd
; P9LE: xxmrghd
; P9LE: xvcvdpsp
; P9LE: xvcvdpsp
; P9LE: vmrgew
; P9LE: xvcvspuxws v2
; P8BE: lfdux
; P8BE: lxsdx
; P8BE: lxsdx
; P8BE: lxsdx
; P8BE: xxmrghd
; P8BE: xxmrghd
; P8BE: xvcvdpsp
; P8BE: xvcvdpsp
; P8BE: vmrgew
; P8BE: xvcvspuxws v2
; P8LE: lfdux
; P8LE: lxsdx
; P8LE: lxsdx
; P8LE: lxsdx
; P8LE: xxmrghd
; P8LE: xxmrghd
; P8LE: xvcvdpsp
; P8LE: xvcvdpsp
; P8LE: vmrgew
; P8LE: xvcvspuxws v2
}

; Function Attrs: norecurse nounwind readnone
define <4 x i32> @spltRegValConvdtoui(double %val) {
entry:
  %conv = fptoui double %val to i32
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %conv, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; P9BE-LABEL: spltRegValConvdtoui
; P9LE-LABEL: spltRegValConvdtoui
; P8BE-LABEL: spltRegValConvdtoui
; P8LE-LABEL: spltRegValConvdtoui
; P9BE: xscvdpuxws
; P9BE: xxspltw
; P9BE: blr
; P9LE: xscvdpuxws
; P9LE: xxspltw
; P9LE: blr
; P8BE: xscvdpuxws
; P8BE: xxspltw
; P8BE: blr
; P8LE: xscvdpuxws
; P8LE: xxspltw
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <4 x i32> @spltMemValConvdtoui(double* nocapture readonly %ptr) {
entry:
  %0 = load double, double* %ptr, align 8
  %conv = fptoui double %0 to i32
  %splat.splatinsert = insertelement <4 x i32> undef, i32 %conv, i32 0
  %splat.splat = shufflevector <4 x i32> %splat.splatinsert, <4 x i32> undef, <4 x i32> zeroinitializer
  ret <4 x i32> %splat.splat
; P9BE-LABEL: spltMemValConvdtoui
; P9LE-LABEL: spltMemValConvdtoui
; P8BE-LABEL: spltMemValConvdtoui
; P8LE-LABEL: spltMemValConvdtoui
; P9BE: lfd
; P9BE: xscvdpuxws
; P9BE: xxspltw
; P9BE: blr
; P9LE: lfd
; P9LE: xscvdpuxws
; P9LE: xxspltw
; P9LE: blr
; P8BE: lfdx
; P8BE: xscvdpuxws
; P8BE: xxspltw
; P8BE: blr
; P8LE: lfdx
; P8LE: xscvdpuxws
; P8LE: xxspltw
; P8LE: blr
}
; Function Attrs: norecurse nounwind readnone
define <2 x i64> @allZeroll() {
entry:
  ret <2 x i64> zeroinitializer
; P9BE-LABEL: allZeroll
; P9LE-LABEL: allZeroll
; P8BE-LABEL: allZeroll
; P8LE-LABEL: allZeroll
; P9BE: xxlxor v2, v2, v2
; P9BE: blr
; P9LE: xxlxor v2, v2, v2
; P9LE: blr
; P8BE: xxlxor v2, v2, v2
; P8BE: blr
; P8LE: xxlxor v2, v2, v2
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @allOnell() {
entry:
  ret <2 x i64> <i64 -1, i64 -1>
; P9BE-LABEL: allOnell
; P9LE-LABEL: allOnell
; P8BE-LABEL: allOnell
; P8LE-LABEL: allOnell
; P9BE: xxspltib v2, 255
; P9BE: blr
; P9LE: xxspltib v2, 255
; P9LE: blr
; P8BE: vspltisb v2, -1
; P8BE: blr
; P8LE: vspltisb v2, -1
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @spltConst1ll() {
entry:
  ret <2 x i64> <i64 1, i64 1>
; P9BE-LABEL: spltConst1ll
; P9LE-LABEL: spltConst1ll
; P8BE-LABEL: spltConst1ll
; P8LE-LABEL: spltConst1ll
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvd2x
; P8BE: blr
; P8LE: lxvd2x
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @spltConst16kll() {
entry:
  ret <2 x i64> <i64 32767, i64 32767>
; P9BE-LABEL: spltConst16kll
; P9LE-LABEL: spltConst16kll
; P8BE-LABEL: spltConst16kll
; P8LE-LABEL: spltConst16kll
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvd2x
; P8BE: blr
; P8LE: lxvd2x
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @spltConst32kll() {
entry:
  ret <2 x i64> <i64 65535, i64 65535>
; P9BE-LABEL: spltConst32kll
; P9LE-LABEL: spltConst32kll
; P8BE-LABEL: spltConst32kll
; P8LE-LABEL: spltConst32kll
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvd2x
; P8BE: blr
; P8LE: lxvd2x
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @fromRegsll(i64 %a, i64 %b) {
entry:
  %vecinit = insertelement <2 x i64> undef, i64 %a, i32 0
  %vecinit1 = insertelement <2 x i64> %vecinit, i64 %b, i32 1
  ret <2 x i64> %vecinit1
; P9BE-LABEL: fromRegsll
; P9LE-LABEL: fromRegsll
; P8BE-LABEL: fromRegsll
; P8LE-LABEL: fromRegsll
; P9BE: mtvsrdd v2, r3, r4
; P9BE: blr
; P9LE: mtvsrdd v2, r4, r3
; P9LE: blr
; P8BE-DAG: mtvsrd {{[vsf0-9]+}}, r3
; P8BE-DAG: mtvsrd {{[vsf0-9]+}}, r4
; P8BE: xxmrghd v2
; P8BE: blr
; P8LE-DAG: mtvsrd {{[vsf0-9]+}}, r3
; P8LE-DAG: mtvsrd {{[vsf0-9]+}}, r4
; P8LE: xxmrghd v2
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @fromDiffConstsll() {
entry:
  ret <2 x i64> <i64 242, i64 -113>
; P9BE-LABEL: fromDiffConstsll
; P9LE-LABEL: fromDiffConstsll
; P8BE-LABEL: fromDiffConstsll
; P8LE-LABEL: fromDiffConstsll
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvd2x
; P8BE: blr
; P8LE: lxvd2x
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemConsAll(i64* nocapture readonly %arr) {
entry:
  %0 = load i64, i64* %arr, align 8
  %vecinit = insertelement <2 x i64> undef, i64 %0, i32 0
  %arrayidx1 = getelementptr inbounds i64, i64* %arr, i64 1
  %1 = load i64, i64* %arrayidx1, align 8
  %vecinit2 = insertelement <2 x i64> %vecinit, i64 %1, i32 1
  ret <2 x i64> %vecinit2
; P9BE-LABEL: fromDiffMemConsAll
; P9LE-LABEL: fromDiffMemConsAll
; P8BE-LABEL: fromDiffMemConsAll
; P8LE-LABEL: fromDiffMemConsAll
; P9BE: lxv v2
; P9BE: blr
; P9LE: lxv v2
; P9LE: blr
; P8BE: lxvd2x v2
; P8BE: blr
; P8LE: lxvd2x
; P8LE: xxswapd v2
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemConsDll(i64* nocapture readonly %arr) {
entry:
  %arrayidx = getelementptr inbounds i64, i64* %arr, i64 3
  %0 = load i64, i64* %arrayidx, align 8
  %vecinit = insertelement <2 x i64> undef, i64 %0, i32 0
  %arrayidx1 = getelementptr inbounds i64, i64* %arr, i64 2
  %1 = load i64, i64* %arrayidx1, align 8
  %vecinit2 = insertelement <2 x i64> %vecinit, i64 %1, i32 1
  ret <2 x i64> %vecinit2
; P9BE-LABEL: fromDiffMemConsDll
; P9LE-LABEL: fromDiffMemConsDll
; P8BE-LABEL: fromDiffMemConsDll
; P8LE-LABEL: fromDiffMemConsDll
; P9BE: lxv v2
; P9BE: blr
; P9LE: lxv
; P9LE: xxswapd v2
; P9LE: blr
; P8BE: lxvd2x
; P8BE: xxswapd v2
; P8BE-NEXT: blr
; P8LE: lxvd2x v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemVarAll(i64* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds i64, i64* %arr, i64 %idxprom
  %0 = load i64, i64* %arrayidx, align 8
  %vecinit = insertelement <2 x i64> undef, i64 %0, i32 0
  %add = add nsw i32 %elem, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i64, i64* %arr, i64 %idxprom1
  %1 = load i64, i64* %arrayidx2, align 8
  %vecinit3 = insertelement <2 x i64> %vecinit, i64 %1, i32 1
  ret <2 x i64> %vecinit3
; P9BE-LABEL: fromDiffMemVarAll
; P9LE-LABEL: fromDiffMemVarAll
; P8BE-LABEL: fromDiffMemVarAll
; P8LE-LABEL: fromDiffMemVarAll
; P9BE: sldi
; P9BE: lxvx v2
; P9BE-NEXT: blr
; P9LE: sldi
; P9LE: lxvx v2
; P9LE-NEXT: blr
; P8BE: sldi
; P8BE: lxvd2x v2
; P8BE-NEXT: blr
; P8LE: sldi
; P8LE: lxvd2x
; P8LE: xxswapd v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemVarDll(i64* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds i64, i64* %arr, i64 %idxprom
  %0 = load i64, i64* %arrayidx, align 8
  %vecinit = insertelement <2 x i64> undef, i64 %0, i32 0
  %sub = add nsw i32 %elem, -1
  %idxprom1 = sext i32 %sub to i64
  %arrayidx2 = getelementptr inbounds i64, i64* %arr, i64 %idxprom1
  %1 = load i64, i64* %arrayidx2, align 8
  %vecinit3 = insertelement <2 x i64> %vecinit, i64 %1, i32 1
  ret <2 x i64> %vecinit3
; P9BE-LABEL: fromDiffMemVarDll
; P9LE-LABEL: fromDiffMemVarDll
; P8BE-LABEL: fromDiffMemVarDll
; P8LE-LABEL: fromDiffMemVarDll
; P9BE: sldi
; P9BE: lxv
; P9BE: xxswapd v2
; P9BE-NEXT: blr
; P9LE: sldi
; P9LE: lxv
; P9LE: xxswapd v2
; P9LE-NEXT: blr
; P8BE: sldi
; P8BE: lxvd2x
; P8BE: xxswapd v2
; P8BE-NEXT: blr
; P8LE: sldi
; P8LE: lxvd2x v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromRandMemConsll(i64* nocapture readonly %arr) {
entry:
  %arrayidx = getelementptr inbounds i64, i64* %arr, i64 4
  %0 = load i64, i64* %arrayidx, align 8
  %vecinit = insertelement <2 x i64> undef, i64 %0, i32 0
  %arrayidx1 = getelementptr inbounds i64, i64* %arr, i64 18
  %1 = load i64, i64* %arrayidx1, align 8
  %vecinit2 = insertelement <2 x i64> %vecinit, i64 %1, i32 1
  ret <2 x i64> %vecinit2
; P9BE-LABEL: fromRandMemConsll
; P9LE-LABEL: fromRandMemConsll
; P8BE-LABEL: fromRandMemConsll
; P8LE-LABEL: fromRandMemConsll
; P9BE: ld
; P9BE: ld
; P9BE: mtvsrdd v2
; P9BE-NEXT: blr
; P9LE: ld
; P9LE: ld
; P9LE: mtvsrdd v2
; P9LE-NEXT: blr
; P8BE: ld
; P8BE: ld
; P8BE-DAG: mtvsrd
; P8BE-DAG: mtvsrd
; P8BE: xxmrghd v2
; P8BE-NEXT: blr
; P8LE: ld
; P8LE: ld
; P8LE-DAG: mtvsrd
; P8LE-DAG: mtvsrd
; P8LE: xxmrghd v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromRandMemVarll(i64* nocapture readonly %arr, i32 signext %elem) {
entry:
  %add = add nsw i32 %elem, 4
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i64, i64* %arr, i64 %idxprom
  %0 = load i64, i64* %arrayidx, align 8
  %vecinit = insertelement <2 x i64> undef, i64 %0, i32 0
  %add1 = add nsw i32 %elem, 1
  %idxprom2 = sext i32 %add1 to i64
  %arrayidx3 = getelementptr inbounds i64, i64* %arr, i64 %idxprom2
  %1 = load i64, i64* %arrayidx3, align 8
  %vecinit4 = insertelement <2 x i64> %vecinit, i64 %1, i32 1
  ret <2 x i64> %vecinit4
; P9BE-LABEL: fromRandMemVarll
; P9LE-LABEL: fromRandMemVarll
; P8BE-LABEL: fromRandMemVarll
; P8LE-LABEL: fromRandMemVarll
; P9BE: sldi
; P9BE: ld
; P9BE: ld
; P9BE: mtvsrdd v2
; P9BE-NEXT: blr
; P9LE: sldi
; P9LE: ld
; P9LE: ld
; P9LE: mtvsrdd v2
; P9LE-NEXT: blr
; P8BE: sldi
; P8BE: ld
; P8BE: ld
; P8BE: mtvsrd
; P8BE: mtvsrd
; P8BE: xxmrghd v2
; P8BE-NEXT: blr
; P8LE: sldi
; P8LE: ld
; P8LE: ld
; P8LE: mtvsrd
; P8LE: mtvsrd
; P8LE: xxmrghd v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @spltRegValll(i64 %val) {
entry:
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %val, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; P9BE-LABEL: spltRegValll
; P9LE-LABEL: spltRegValll
; P8BE-LABEL: spltRegValll
; P8LE-LABEL: spltRegValll
; P9BE: mtvsrdd v2, r3, r3
; P9BE-NEXT: blr
; P9LE: mtvsrdd v2, r3, r3
; P9LE-NEXT: blr
; P8BE: mtvsrd {{[vsf]+}}[[REG1:[0-9]+]], r3
; P8BE: xxspltd v2, {{[vsf]+}}[[REG1]], 0
; P8BE-NEXT: blr
; P8LE: mtvsrd {{[vsf]+}}[[REG1:[0-9]+]], r3
; P8LE: xxspltd v2, {{[vsf]+}}[[REG1]], 0
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @spltMemValll(i64* nocapture readonly %ptr) {
entry:
  %0 = load i64, i64* %ptr, align 8
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %0, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; P9BE-LABEL: spltMemValll
; P9LE-LABEL: spltMemValll
; P8BE-LABEL: spltMemValll
; P8LE-LABEL: spltMemValll
; P9BE: lxvdsx v2
; P9BE-NEXT: blr
; P9LE: lxvdsx v2
; P9LE-NEXT: blr
; P8BE: lxvdsx v2
; P8BE-NEXT: blr
; P8LE: lxvdsx v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @spltCnstConvftoll() {
entry:
  ret <2 x i64> <i64 4, i64 4>
; P9BE-LABEL: spltCnstConvftoll
; P9LE-LABEL: spltCnstConvftoll
; P8BE-LABEL: spltCnstConvftoll
; P8LE-LABEL: spltCnstConvftoll
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvd2x
; P8BE: blr
; P8LE: lxvd2x
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @fromRegsConvftoll(float %a, float %b) {
entry:
  %conv = fptosi float %a to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %conv1 = fptosi float %b to i64
  %vecinit2 = insertelement <2 x i64> %vecinit, i64 %conv1, i32 1
  ret <2 x i64> %vecinit2
; P9BE-LABEL: fromRegsConvftoll
; P9LE-LABEL: fromRegsConvftoll
; P8BE-LABEL: fromRegsConvftoll
; P8LE-LABEL: fromRegsConvftoll
; P9BE: xxmrghd
; P9BE: xvcvdpsxds v2
; P9BE-NEXT: blr
; P9LE: xxmrghd
; P9LE: xvcvdpsxds v2
; P9LE-NEXT: blr
; P8BE: xxmrghd
; P8BE: xvcvdpsxds v2
; P8BE-NEXT: blr
; P8LE: xxmrghd
; P8LE: xvcvdpsxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @fromDiffConstsConvftoll() {
entry:
  ret <2 x i64> <i64 24, i64 234>
; P9BE-LABEL: fromDiffConstsConvftoll
; P9LE-LABEL: fromDiffConstsConvftoll
; P8BE-LABEL: fromDiffConstsConvftoll
; P8LE-LABEL: fromDiffConstsConvftoll
; P9BE: lxvx v2
; P9BE: blr
; P9LE: lxvx v2
; P9LE: blr
; P8BE: lxvd2x v2
; P8BE: blr
; P8LE: lxvd2x
; P8LE: xxswapd v2
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemConsAConvftoll(float* nocapture readonly %ptr) {
entry:
  %0 = load float, float* %ptr, align 4
  %conv = fptosi float %0 to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %arrayidx1 = getelementptr inbounds float, float* %ptr, i64 1
  %1 = load float, float* %arrayidx1, align 4
  %conv2 = fptosi float %1 to i64
  %vecinit3 = insertelement <2 x i64> %vecinit, i64 %conv2, i32 1
  ret <2 x i64> %vecinit3
; P9BE-LABEL: fromDiffMemConsAConvftoll
; P9LE-LABEL: fromDiffMemConsAConvftoll
; P8BE-LABEL: fromDiffMemConsAConvftoll
; P8LE-LABEL: fromDiffMemConsAConvftoll
; P9BE: lfs
; P9BE: lfs
; P9BE: xxmrghd
; P9BE-NEXT: xvcvdpsxds v2
; P9BE-NEXT: blr
; P9LE: lfs
; P9LE: lfs
; P9LE: xxmrghd
; P9LE-NEXT: xvcvdpsxds v2
; P9LE-NEXT: blr
; P8BE: lfsx
; P8BE: lfsx
; P8BE: xxmrghd
; P8BE-NEXT: xvcvdpsxds v2
; P8BE-NEXT: blr
; P8LE: lfsx
; P8LE: lfsx
; P8LE: xxmrghd
; P8LE-NEXT: xvcvdpsxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemConsDConvftoll(float* nocapture readonly %ptr) {
entry:
  %arrayidx = getelementptr inbounds float, float* %ptr, i64 3
  %0 = load float, float* %arrayidx, align 4
  %conv = fptosi float %0 to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %arrayidx1 = getelementptr inbounds float, float* %ptr, i64 2
  %1 = load float, float* %arrayidx1, align 4
  %conv2 = fptosi float %1 to i64
  %vecinit3 = insertelement <2 x i64> %vecinit, i64 %conv2, i32 1
  ret <2 x i64> %vecinit3
; P9BE-LABEL: fromDiffMemConsDConvftoll
; P9LE-LABEL: fromDiffMemConsDConvftoll
; P8BE-LABEL: fromDiffMemConsDConvftoll
; P8LE-LABEL: fromDiffMemConsDConvftoll
; P9BE: lfs
; P9BE: lfs
; P9BE: xxmrghd
; P9BE-NEXT: xvcvdpsxds v2
; P9BE-NEXT: blr
; P9LE: lfs
; P9LE: lfs
; P9LE: xxmrghd
; P9LE-NEXT: xvcvdpsxds v2
; P9LE-NEXT: blr
; P8BE: lfsx
; P8BE: lfsx
; P8BE: xxmrghd
; P8BE-NEXT: xvcvdpsxds v2
; P8BE-NEXT: blr
; P8LE: lfsx
; P8LE: lfsx
; P8LE: xxmrghd
; P8LE-NEXT: xvcvdpsxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemVarAConvftoll(float* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds float, float* %arr, i64 %idxprom
  %0 = load float, float* %arrayidx, align 4
  %conv = fptosi float %0 to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %add = add nsw i32 %elem, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds float, float* %arr, i64 %idxprom1
  %1 = load float, float* %arrayidx2, align 4
  %conv3 = fptosi float %1 to i64
  %vecinit4 = insertelement <2 x i64> %vecinit, i64 %conv3, i32 1
  ret <2 x i64> %vecinit4
; P9BE-LABEL: fromDiffMemVarAConvftoll
; P9LE-LABEL: fromDiffMemVarAConvftoll
; P8BE-LABEL: fromDiffMemVarAConvftoll
; P8LE-LABEL: fromDiffMemVarAConvftoll
; P9BE: sldi
; P9BE: lfsux
; P9BE: lfs
; P9BE: xxmrghd
; P9BE-NEXT: xvcvdpsxds v2
; P9BE-NEXT: blr
; P9LE: sldi
; P9LE: lfsux
; P9LE: lfs
; P9LE: xxmrghd
; P9LE-NEXT: xvcvdpsxds v2
; P9LE-NEXT: blr
; P8BE: sldi
; P8BE: lfsux
; P8BE: lfsx
; P8BE: xxmrghd
; P8BE-NEXT: xvcvdpsxds v2
; P8BE-NEXT: blr
; P8LE: sldi
; P8LE: lfsux
; P8LE: lfsx
; P8LE: xxmrghd
; P8LE-NEXT: xvcvdpsxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemVarDConvftoll(float* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds float, float* %arr, i64 %idxprom
  %0 = load float, float* %arrayidx, align 4
  %conv = fptosi float %0 to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %sub = add nsw i32 %elem, -1
  %idxprom1 = sext i32 %sub to i64
  %arrayidx2 = getelementptr inbounds float, float* %arr, i64 %idxprom1
  %1 = load float, float* %arrayidx2, align 4
  %conv3 = fptosi float %1 to i64
  %vecinit4 = insertelement <2 x i64> %vecinit, i64 %conv3, i32 1
  ret <2 x i64> %vecinit4
; P9BE-LABEL: fromDiffMemVarDConvftoll
; P9LE-LABEL: fromDiffMemVarDConvftoll
; P8BE-LABEL: fromDiffMemVarDConvftoll
; P8LE-LABEL: fromDiffMemVarDConvftoll
; P9BE: sldi
; P9BE: lfsux
; P9BE: lfs
; P9BE: xxmrghd
; P9BE-NEXT: xvcvdpsxds v2
; P9BE-NEXT: blr
; P9LE: sldi
; P9LE: lfsux
; P9LE: lfs
; P9LE: xxmrghd
; P9LE-NEXT: xvcvdpsxds v2
; P9LE-NEXT: blr
; P8BE: sldi
; P8BE: lfsux
; P8BE: lfsx
; P8BE: xxmrghd
; P8BE-NEXT: xvcvdpsxds v2
; P8BE-NEXT: blr
; P8LE: sldi
; P8LE: lfsux
; P8LE: lfsx
; P8LE: xxmrghd
; P8LE-NEXT: xvcvdpsxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @spltRegValConvftoll(float %val) {
entry:
  %conv = fptosi float %val to i64
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %conv, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; P9BE-LABEL: spltRegValConvftoll
; P9LE-LABEL: spltRegValConvftoll
; P8BE-LABEL: spltRegValConvftoll
; P8LE-LABEL: spltRegValConvftoll
; P9BE: xscvdpsxds
; P9BE-NEXT: xxspltd v2
; P9BE-NEXT: blr
; P9LE: xscvdpsxds
; P9LE-NEXT: xxspltd v2
; P9LE-NEXT: blr
; P8BE: xscvdpsxds
; P8BE-NEXT: xxspltd v2
; P8BE-NEXT: blr
; P8LE: xscvdpsxds
; P8LE-NEXT: xxspltd v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @spltMemValConvftoll(float* nocapture readonly %ptr) {
entry:
  %0 = load float, float* %ptr, align 4
  %conv = fptosi float %0 to i64
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %conv, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; P9BE-LABEL: spltMemValConvftoll
; P9LE-LABEL: spltMemValConvftoll
; P8BE-LABEL: spltMemValConvftoll
; P8LE-LABEL: spltMemValConvftoll
; P9BE: lfs
; P9BE-NEXT: xscvdpsxds
; P9BE-NEXT: xxspltd v2
; P9BE-NEXT: blr
; P9LE: lfs
; P9LE-NEXT: xscvdpsxds
; P9LE-NEXT: xxspltd v2
; P9LE-NEXT: blr
; P8BE: lfsx
; P8BE-NEXT: xscvdpsxds
; P8BE-NEXT: xxspltd v2
; P8BE-NEXT: blr
; P8LE: lfsx
; P8LE-NEXT: xscvdpsxds
; P8LE-NEXT: xxspltd v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @spltCnstConvdtoll() {
entry:
  ret <2 x i64> <i64 4, i64 4>
; P9BE-LABEL: spltCnstConvdtoll
; P9LE-LABEL: spltCnstConvdtoll
; P8BE-LABEL: spltCnstConvdtoll
; P8LE-LABEL: spltCnstConvdtoll
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvd2x
; P8BE: blr
; P8LE: lxvd2x
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @fromRegsConvdtoll(double %a, double %b) {
entry:
  %conv = fptosi double %a to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %conv1 = fptosi double %b to i64
  %vecinit2 = insertelement <2 x i64> %vecinit, i64 %conv1, i32 1
  ret <2 x i64> %vecinit2
; P9BE-LABEL: fromRegsConvdtoll
; P9LE-LABEL: fromRegsConvdtoll
; P8BE-LABEL: fromRegsConvdtoll
; P8LE-LABEL: fromRegsConvdtoll
; P9BE: xxmrghd
; P9BE-NEXT: xvcvdpsxds
; P9BE-NEXT: blr
; P9LE: xxmrghd
; P9LE-NEXT: xvcvdpsxds
; P9LE-NEXT: blr
; P8BE: xxmrghd
; P8BE-NEXT: xvcvdpsxds
; P8BE-NEXT: blr
; P8LE: xxmrghd
; P8LE-NEXT: xvcvdpsxds
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @fromDiffConstsConvdtoll() {
entry:
  ret <2 x i64> <i64 24, i64 234>
; P9BE-LABEL: fromDiffConstsConvdtoll
; P9LE-LABEL: fromDiffConstsConvdtoll
; P8BE-LABEL: fromDiffConstsConvdtoll
; P8LE-LABEL: fromDiffConstsConvdtoll
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvd2x
; P8BE: blr
; P8LE: lxvd2x
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemConsAConvdtoll(double* nocapture readonly %ptr) {
entry:
  %0 = bitcast double* %ptr to <2 x double>*
  %1 = load <2 x double>, <2 x double>* %0, align 8
  %2 = fptosi <2 x double> %1 to <2 x i64>
  ret <2 x i64> %2
; P9BE-LABEL: fromDiffMemConsAConvdtoll
; P9LE-LABEL: fromDiffMemConsAConvdtoll
; P8BE-LABEL: fromDiffMemConsAConvdtoll
; P8LE-LABEL: fromDiffMemConsAConvdtoll
; P9BE: lxv
; P9BE-NEXT: xvcvdpsxds v2
; P9BE-NEXT: blr
; P9LE: lxv
; P9LE-NEXT: xvcvdpsxds v2
; P9LE-NEXT: blr
; P8BE: lxvd2x
; P8BE-NEXT: xvcvdpsxds v2
; P8BE-NEXT: blr
; P8LE: lxvd2x
; P8LE: xxswapd
; P8LE-NEXT: xvcvdpsxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemConsDConvdtoll(double* nocapture readonly %ptr) {
entry:
  %arrayidx = getelementptr inbounds double, double* %ptr, i64 3
  %0 = load double, double* %arrayidx, align 8
  %conv = fptosi double %0 to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %arrayidx1 = getelementptr inbounds double, double* %ptr, i64 2
  %1 = load double, double* %arrayidx1, align 8
  %conv2 = fptosi double %1 to i64
  %vecinit3 = insertelement <2 x i64> %vecinit, i64 %conv2, i32 1
  ret <2 x i64> %vecinit3
; P9BE-LABEL: fromDiffMemConsDConvdtoll
; P9LE-LABEL: fromDiffMemConsDConvdtoll
; P8BE-LABEL: fromDiffMemConsDConvdtoll
; P8LE-LABEL: fromDiffMemConsDConvdtoll
; P9BE: lxv
; P9BE-NEXT: xxswapd
; P9BE-NEXT: xvcvdpsxds v2
; P9BE-NEXT: blr
; P9LE: lxv
; P9LE-NEXT: xxswapd
; P9LE-NEXT: xvcvdpsxds v2
; P9LE-NEXT: blr
; P8BE: lxvd2x
; P8BE-NEXT: xxswapd
; P8BE-NEXT: xvcvdpsxds v2
; P8BE-NEXT: blr
; P8LE: lxvd2x
; P8LE-NEXT: xvcvdpsxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemVarAConvdtoll(double* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds double, double* %arr, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
  %conv = fptosi double %0 to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %add = add nsw i32 %elem, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds double, double* %arr, i64 %idxprom1
  %1 = load double, double* %arrayidx2, align 8
  %conv3 = fptosi double %1 to i64
  %vecinit4 = insertelement <2 x i64> %vecinit, i64 %conv3, i32 1
  ret <2 x i64> %vecinit4
; P9BE-LABEL: fromDiffMemVarAConvdtoll
; P9LE-LABEL: fromDiffMemVarAConvdtoll
; P8BE-LABEL: fromDiffMemVarAConvdtoll
; P8LE-LABEL: fromDiffMemVarAConvdtoll
; P9BE: sldi
; P9BE: lxvx
; P9BE-NEXT: xvcvdpsxds v2
; P9BE-NEXT: blr
; P9LE: sldi
; P9LE: lxvx
; P9LE-NEXT: xvcvdpsxds v2
; P9LE-NEXT: blr
; P8BE: sldi
; P8BE: lxvd2x
; P8BE-NEXT: xvcvdpsxds v2
; P8BE-NEXT: blr
; P8LE: sldi
; P8LE: lxvd2x
; P8LE-NEXT: xxswapd
; P8LE-NEXT: xvcvdpsxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemVarDConvdtoll(double* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds double, double* %arr, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
  %conv = fptosi double %0 to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %sub = add nsw i32 %elem, -1
  %idxprom1 = sext i32 %sub to i64
  %arrayidx2 = getelementptr inbounds double, double* %arr, i64 %idxprom1
  %1 = load double, double* %arrayidx2, align 8
  %conv3 = fptosi double %1 to i64
  %vecinit4 = insertelement <2 x i64> %vecinit, i64 %conv3, i32 1
  ret <2 x i64> %vecinit4
; P9BE-LABEL: fromDiffMemVarDConvdtoll
; P9LE-LABEL: fromDiffMemVarDConvdtoll
; P8BE-LABEL: fromDiffMemVarDConvdtoll
; P8LE-LABEL: fromDiffMemVarDConvdtoll
; P9BE: sldi
; P9BE: lxv
; P9BE-NEXT: xxswapd
; P9BE-NEXT: xvcvdpsxds v2
; P9BE-NEXT: blr
; P9LE: sldi
; P9LE: lxv
; P9LE-NEXT: xxswapd
; P9LE-NEXT: xvcvdpsxds v2
; P9LE-NEXT: blr
; P8BE: sldi
; P8BE: lxvd2x
; P8BE-NEXT: xxswapd
; P8BE-NEXT: xvcvdpsxds v2
; P8BE-NEXT: blr
; P8LE: sldi
; P8LE: lxvd2x
; P8LE-NEXT: xvcvdpsxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @spltRegValConvdtoll(double %val) {
entry:
  %conv = fptosi double %val to i64
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %conv, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; P9BE-LABEL: spltRegValConvdtoll
; P9LE-LABEL: spltRegValConvdtoll
; P8BE-LABEL: spltRegValConvdtoll
; P8LE-LABEL: spltRegValConvdtoll
; P9BE: xscvdpsxds
; P9BE-NEXT: xxspltd v2
; P9BE-NEXT: blr
; P9LE: xscvdpsxds
; P9LE-NEXT: xxspltd v2
; P9LE-NEXT: blr
; P8BE: xscvdpsxds
; P8BE-NEXT: xxspltd v2
; P8BE-NEXT: blr
; P8LE: xscvdpsxds
; P8LE-NEXT: xxspltd v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @spltMemValConvdtoll(double* nocapture readonly %ptr) {
entry:
  %0 = load double, double* %ptr, align 8
  %conv = fptosi double %0 to i64
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %conv, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; P9BE-LABEL: spltMemValConvdtoll
; P9LE-LABEL: spltMemValConvdtoll
; P8BE-LABEL: spltMemValConvdtoll
; P8LE-LABEL: spltMemValConvdtoll
; P9BE: lxvdsx
; P9BE-NEXT: xvcvdpsxds
; P9BE-NEXT: blr
; P9LE: lxvdsx
; P9LE-NEXT: xvcvdpsxds
; P9LE-NEXT: blr
; P8BE: lxvdsx
; P8BE-NEXT: xvcvdpsxds
; P8BE-NEXT: blr
; P8LE: lxvdsx
; P8LE-NEXT: xvcvdpsxds
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @allZeroull() {
entry:
  ret <2 x i64> zeroinitializer
; P9BE-LABEL: allZeroull
; P9LE-LABEL: allZeroull
; P8BE-LABEL: allZeroull
; P8LE-LABEL: allZeroull
; P9BE: xxlxor v2, v2, v2
; P9BE: blr
; P9LE: xxlxor v2, v2, v2
; P9LE: blr
; P8BE: xxlxor v2, v2, v2
; P8BE: blr
; P8LE: xxlxor v2, v2, v2
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @allOneull() {
entry:
  ret <2 x i64> <i64 -1, i64 -1>
; P9BE-LABEL: allOneull
; P9LE-LABEL: allOneull
; P8BE-LABEL: allOneull
; P8LE-LABEL: allOneull
; P9BE: xxspltib v2, 255
; P9BE: blr
; P9LE: xxspltib v2, 255
; P9LE: blr
; P8BE: vspltisb v2, -1
; P8BE: blr
; P8LE: vspltisb v2, -1
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @spltConst1ull() {
entry:
  ret <2 x i64> <i64 1, i64 1>
; P9BE-LABEL: spltConst1ull
; P9LE-LABEL: spltConst1ull
; P8BE-LABEL: spltConst1ull
; P8LE-LABEL: spltConst1ull
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvd2x
; P8BE: blr
; P8LE: lxvd2x
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @spltConst16kull() {
entry:
  ret <2 x i64> <i64 32767, i64 32767>
; P9BE-LABEL: spltConst16kull
; P9LE-LABEL: spltConst16kull
; P8BE-LABEL: spltConst16kull
; P8LE-LABEL: spltConst16kull
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvd2x
; P8BE: blr
; P8LE: lxvd2x
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @spltConst32kull() {
entry:
  ret <2 x i64> <i64 65535, i64 65535>
; P9BE-LABEL: spltConst32kull
; P9LE-LABEL: spltConst32kull
; P8BE-LABEL: spltConst32kull
; P8LE-LABEL: spltConst32kull
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvd2x
; P8BE: blr
; P8LE: lxvd2x
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @fromRegsull(i64 %a, i64 %b) {
entry:
  %vecinit = insertelement <2 x i64> undef, i64 %a, i32 0
  %vecinit1 = insertelement <2 x i64> %vecinit, i64 %b, i32 1
  ret <2 x i64> %vecinit1
; P9BE-LABEL: fromRegsull
; P9LE-LABEL: fromRegsull
; P8BE-LABEL: fromRegsull
; P8LE-LABEL: fromRegsull
; P9BE: mtvsrdd v2, r3, r4
; P9BE: blr
; P9LE: mtvsrdd v2, r4, r3
; P9LE: blr
; P8BE-DAG: mtvsrd {{[vsf0-9]+}}, r3
; P8BE-DAG: mtvsrd {{[vsf0-9]+}}, r4
; P8BE: xxmrghd v2
; P8BE: blr
; P8LE-DAG: mtvsrd {{[vsf0-9]+}}, r3
; P8LE-DAG: mtvsrd {{[vsf0-9]+}}, r4
; P8LE: xxmrghd v2
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @fromDiffConstsull() {
entry:
  ret <2 x i64> <i64 242, i64 -113>
; P9BE-LABEL: fromDiffConstsull
; P9LE-LABEL: fromDiffConstsull
; P8BE-LABEL: fromDiffConstsull
; P8LE-LABEL: fromDiffConstsull
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvd2x
; P8BE: blr
; P8LE: lxvd2x
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemConsAull(i64* nocapture readonly %arr) {
entry:
  %0 = load i64, i64* %arr, align 8
  %vecinit = insertelement <2 x i64> undef, i64 %0, i32 0
  %arrayidx1 = getelementptr inbounds i64, i64* %arr, i64 1
  %1 = load i64, i64* %arrayidx1, align 8
  %vecinit2 = insertelement <2 x i64> %vecinit, i64 %1, i32 1
  ret <2 x i64> %vecinit2
; P9BE-LABEL: fromDiffMemConsAull
; P9LE-LABEL: fromDiffMemConsAull
; P8BE-LABEL: fromDiffMemConsAull
; P8LE-LABEL: fromDiffMemConsAull
; P9BE: lxv v2
; P9BE: blr
; P9LE: lxv v2
; P9LE: blr
; P8BE: lxvd2x v2
; P8BE: blr
; P8LE: lxvd2x
; P8LE: xxswapd v2
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemConsDull(i64* nocapture readonly %arr) {
entry:
  %arrayidx = getelementptr inbounds i64, i64* %arr, i64 3
  %0 = load i64, i64* %arrayidx, align 8
  %vecinit = insertelement <2 x i64> undef, i64 %0, i32 0
  %arrayidx1 = getelementptr inbounds i64, i64* %arr, i64 2
  %1 = load i64, i64* %arrayidx1, align 8
  %vecinit2 = insertelement <2 x i64> %vecinit, i64 %1, i32 1
  ret <2 x i64> %vecinit2
; P9BE-LABEL: fromDiffMemConsDull
; P9LE-LABEL: fromDiffMemConsDull
; P8BE-LABEL: fromDiffMemConsDull
; P8LE-LABEL: fromDiffMemConsDull
; P9BE: lxv v2
; P9BE: blr
; P9LE: lxv
; P9LE: xxswapd v2
; P9LE: blr
; P8BE: lxvd2x
; P8BE: xxswapd v2
; P8BE-NEXT: blr
; P8LE: lxvd2x v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemVarAull(i64* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds i64, i64* %arr, i64 %idxprom
  %0 = load i64, i64* %arrayidx, align 8
  %vecinit = insertelement <2 x i64> undef, i64 %0, i32 0
  %add = add nsw i32 %elem, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds i64, i64* %arr, i64 %idxprom1
  %1 = load i64, i64* %arrayidx2, align 8
  %vecinit3 = insertelement <2 x i64> %vecinit, i64 %1, i32 1
  ret <2 x i64> %vecinit3
; P9BE-LABEL: fromDiffMemVarAull
; P9LE-LABEL: fromDiffMemVarAull
; P8BE-LABEL: fromDiffMemVarAull
; P8LE-LABEL: fromDiffMemVarAull
; P9BE: sldi
; P9BE: lxvx v2
; P9BE-NEXT: blr
; P9LE: sldi
; P9LE: lxvx v2
; P9LE-NEXT: blr
; P8BE: sldi
; P8BE: lxvd2x v2
; P8BE-NEXT: blr
; P8LE: sldi
; P8LE: lxvd2x
; P8LE: xxswapd v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemVarDull(i64* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds i64, i64* %arr, i64 %idxprom
  %0 = load i64, i64* %arrayidx, align 8
  %vecinit = insertelement <2 x i64> undef, i64 %0, i32 0
  %sub = add nsw i32 %elem, -1
  %idxprom1 = sext i32 %sub to i64
  %arrayidx2 = getelementptr inbounds i64, i64* %arr, i64 %idxprom1
  %1 = load i64, i64* %arrayidx2, align 8
  %vecinit3 = insertelement <2 x i64> %vecinit, i64 %1, i32 1
  ret <2 x i64> %vecinit3
; P9BE-LABEL: fromDiffMemVarDull
; P9LE-LABEL: fromDiffMemVarDull
; P8BE-LABEL: fromDiffMemVarDull
; P8LE-LABEL: fromDiffMemVarDull
; P9BE: sldi
; P9BE: lxv
; P9BE: xxswapd v2
; P9BE-NEXT: blr
; P9LE: sldi
; P9LE: lxv
; P9LE: xxswapd v2
; P9LE-NEXT: blr
; P8BE: sldi
; P8BE: lxvd2x
; P8BE: xxswapd v2
; P8BE-NEXT: blr
; P8LE: sldi
; P8LE: lxvd2x v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromRandMemConsull(i64* nocapture readonly %arr) {
entry:
  %arrayidx = getelementptr inbounds i64, i64* %arr, i64 4
  %0 = load i64, i64* %arrayidx, align 8
  %vecinit = insertelement <2 x i64> undef, i64 %0, i32 0
  %arrayidx1 = getelementptr inbounds i64, i64* %arr, i64 18
  %1 = load i64, i64* %arrayidx1, align 8
  %vecinit2 = insertelement <2 x i64> %vecinit, i64 %1, i32 1
  ret <2 x i64> %vecinit2
; P9BE-LABEL: fromRandMemConsull
; P9LE-LABEL: fromRandMemConsull
; P8BE-LABEL: fromRandMemConsull
; P8LE-LABEL: fromRandMemConsull
; P9BE: ld
; P9BE: ld
; P9BE: mtvsrdd v2
; P9BE-NEXT: blr
; P9LE: ld
; P9LE: ld
; P9LE: mtvsrdd v2
; P9LE-NEXT: blr
; P8BE: ld
; P8BE: ld
; P8BE-DAG: mtvsrd
; P8BE-DAG: mtvsrd
; P8BE: xxmrghd v2
; P8BE-NEXT: blr
; P8LE: ld
; P8LE: ld
; P8LE-DAG: mtvsrd
; P8LE-DAG: mtvsrd
; P8LE: xxmrghd v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromRandMemVarull(i64* nocapture readonly %arr, i32 signext %elem) {
entry:
  %add = add nsw i32 %elem, 4
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i64, i64* %arr, i64 %idxprom
  %0 = load i64, i64* %arrayidx, align 8
  %vecinit = insertelement <2 x i64> undef, i64 %0, i32 0
  %add1 = add nsw i32 %elem, 1
  %idxprom2 = sext i32 %add1 to i64
  %arrayidx3 = getelementptr inbounds i64, i64* %arr, i64 %idxprom2
  %1 = load i64, i64* %arrayidx3, align 8
  %vecinit4 = insertelement <2 x i64> %vecinit, i64 %1, i32 1
  ret <2 x i64> %vecinit4
; P9BE-LABEL: fromRandMemVarull
; P9LE-LABEL: fromRandMemVarull
; P8BE-LABEL: fromRandMemVarull
; P8LE-LABEL: fromRandMemVarull
; P9BE: sldi
; P9BE: ld
; P9BE: ld
; P9BE: mtvsrdd v2
; P9BE-NEXT: blr
; P9LE: sldi
; P9LE: ld
; P9LE: ld
; P9LE: mtvsrdd v2
; P9LE-NEXT: blr
; P8BE: sldi
; P8BE: ld
; P8BE: ld
; P8BE: mtvsrd
; P8BE: mtvsrd
; P8BE: xxmrghd v2
; P8BE-NEXT: blr
; P8LE: sldi
; P8LE: ld
; P8LE: ld
; P8LE: mtvsrd
; P8LE: mtvsrd
; P8LE: xxmrghd v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @spltRegValull(i64 %val) {
entry:
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %val, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; P9BE-LABEL: spltRegValull
; P9LE-LABEL: spltRegValull
; P8BE-LABEL: spltRegValull
; P8LE-LABEL: spltRegValull
; P9BE: mtvsrdd v2, r3, r3
; P9BE-NEXT: blr
; P9LE: mtvsrdd v2, r3, r3
; P9LE-NEXT: blr
; P8BE: mtvsrd {{[vsf]+}}[[REG1:[0-9]+]], r3
; P8BE: xxspltd v2, {{[vsf]+}}[[REG1]], 0
; P8BE-NEXT: blr
; P8LE: mtvsrd {{[vsf]+}}[[REG1:[0-9]+]], r3
; P8LE: xxspltd v2, {{[vsf]+}}[[REG1]], 0
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @spltMemValull(i64* nocapture readonly %ptr) {
entry:
  %0 = load i64, i64* %ptr, align 8
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %0, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; P9BE-LABEL: spltMemValull
; P9LE-LABEL: spltMemValull
; P8BE-LABEL: spltMemValull
; P8LE-LABEL: spltMemValull
; P9BE: lxvdsx v2
; P9BE-NEXT: blr
; P9LE: lxvdsx v2
; P9LE-NEXT: blr
; P8BE: lxvdsx v2
; P8BE-NEXT: blr
; P8LE: lxvdsx v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @spltCnstConvftoull() {
entry:
  ret <2 x i64> <i64 4, i64 4>
; P9BE-LABEL: spltCnstConvftoull
; P9LE-LABEL: spltCnstConvftoull
; P8BE-LABEL: spltCnstConvftoull
; P8LE-LABEL: spltCnstConvftoull
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvd2x
; P8BE: blr
; P8LE: lxvd2x
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @fromRegsConvftoull(float %a, float %b) {
entry:
  %conv = fptoui float %a to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %conv1 = fptoui float %b to i64
  %vecinit2 = insertelement <2 x i64> %vecinit, i64 %conv1, i32 1
  ret <2 x i64> %vecinit2
; P9BE-LABEL: fromRegsConvftoull
; P9LE-LABEL: fromRegsConvftoull
; P8BE-LABEL: fromRegsConvftoull
; P8LE-LABEL: fromRegsConvftoull
; P9BE: xxmrghd
; P9BE: xvcvdpuxds v2
; P9BE-NEXT: blr
; P9LE: xxmrghd
; P9LE: xvcvdpuxds v2
; P9LE-NEXT: blr
; P8BE: xxmrghd
; P8BE: xvcvdpuxds v2
; P8BE-NEXT: blr
; P8LE: xxmrghd
; P8LE: xvcvdpuxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @fromDiffConstsConvftoull() {
entry:
  ret <2 x i64> <i64 24, i64 234>
; P9BE-LABEL: fromDiffConstsConvftoull
; P9LE-LABEL: fromDiffConstsConvftoull
; P8BE-LABEL: fromDiffConstsConvftoull
; P8LE-LABEL: fromDiffConstsConvftoull
; P9BE: lxvx v2
; P9BE: blr
; P9LE: lxvx v2
; P9LE: blr
; P8BE: lxvd2x v2
; P8BE: blr
; P8LE: lxvd2x
; P8LE: xxswapd v2
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemConsAConvftoull(float* nocapture readonly %ptr) {
entry:
  %0 = load float, float* %ptr, align 4
  %conv = fptoui float %0 to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %arrayidx1 = getelementptr inbounds float, float* %ptr, i64 1
  %1 = load float, float* %arrayidx1, align 4
  %conv2 = fptoui float %1 to i64
  %vecinit3 = insertelement <2 x i64> %vecinit, i64 %conv2, i32 1
  ret <2 x i64> %vecinit3
; P9BE-LABEL: fromDiffMemConsAConvftoull
; P9LE-LABEL: fromDiffMemConsAConvftoull
; P8BE-LABEL: fromDiffMemConsAConvftoull
; P8LE-LABEL: fromDiffMemConsAConvftoull
; P9BE: lfs
; P9BE: lfs
; P9BE: xxmrghd
; P9BE-NEXT: xvcvdpuxds v2
; P9BE-NEXT: blr
; P9LE: lfs
; P9LE: lfs
; P9LE: xxmrghd
; P9LE-NEXT: xvcvdpuxds v2
; P9LE-NEXT: blr
; P8BE: lfsx
; P8BE: lfsx
; P8BE: xxmrghd
; P8BE-NEXT: xvcvdpuxds v2
; P8BE-NEXT: blr
; P8LE: lfsx
; P8LE: lfsx
; P8LE: xxmrghd
; P8LE-NEXT: xvcvdpuxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemConsDConvftoull(float* nocapture readonly %ptr) {
entry:
  %arrayidx = getelementptr inbounds float, float* %ptr, i64 3
  %0 = load float, float* %arrayidx, align 4
  %conv = fptoui float %0 to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %arrayidx1 = getelementptr inbounds float, float* %ptr, i64 2
  %1 = load float, float* %arrayidx1, align 4
  %conv2 = fptoui float %1 to i64
  %vecinit3 = insertelement <2 x i64> %vecinit, i64 %conv2, i32 1
  ret <2 x i64> %vecinit3
; P9BE-LABEL: fromDiffMemConsDConvftoull
; P9LE-LABEL: fromDiffMemConsDConvftoull
; P8BE-LABEL: fromDiffMemConsDConvftoull
; P8LE-LABEL: fromDiffMemConsDConvftoull
; P9BE: lfs
; P9BE: lfs
; P9BE: xxmrghd
; P9BE-NEXT: xvcvdpuxds v2
; P9BE-NEXT: blr
; P9LE: lfs
; P9LE: lfs
; P9LE: xxmrghd
; P9LE-NEXT: xvcvdpuxds v2
; P9LE-NEXT: blr
; P8BE: lfsx
; P8BE: lfsx
; P8BE: xxmrghd
; P8BE-NEXT: xvcvdpuxds v2
; P8BE-NEXT: blr
; P8LE: lfsx
; P8LE: lfsx
; P8LE: xxmrghd
; P8LE-NEXT: xvcvdpuxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemVarAConvftoull(float* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds float, float* %arr, i64 %idxprom
  %0 = load float, float* %arrayidx, align 4
  %conv = fptoui float %0 to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %add = add nsw i32 %elem, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds float, float* %arr, i64 %idxprom1
  %1 = load float, float* %arrayidx2, align 4
  %conv3 = fptoui float %1 to i64
  %vecinit4 = insertelement <2 x i64> %vecinit, i64 %conv3, i32 1
  ret <2 x i64> %vecinit4
; P9BE-LABEL: fromDiffMemVarAConvftoull
; P9LE-LABEL: fromDiffMemVarAConvftoull
; P8BE-LABEL: fromDiffMemVarAConvftoull
; P8LE-LABEL: fromDiffMemVarAConvftoull
; P9BE: sldi
; P9BE: lfsux
; P9BE: lfs
; P9BE: xxmrghd
; P9BE-NEXT: xvcvdpuxds v2
; P9BE-NEXT: blr
; P9LE: sldi
; P9LE: lfsux
; P9LE: lfs
; P9LE: xxmrghd
; P9LE-NEXT: xvcvdpuxds v2
; P9LE-NEXT: blr
; P8BE: sldi
; P8BE: lfsux
; P8BE: lfsx
; P8BE: xxmrghd
; P8BE-NEXT: xvcvdpuxds v2
; P8BE-NEXT: blr
; P8LE: sldi
; P8LE: lfsux
; P8LE: lfsx
; P8LE: xxmrghd
; P8LE-NEXT: xvcvdpuxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemVarDConvftoull(float* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds float, float* %arr, i64 %idxprom
  %0 = load float, float* %arrayidx, align 4
  %conv = fptoui float %0 to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %sub = add nsw i32 %elem, -1
  %idxprom1 = sext i32 %sub to i64
  %arrayidx2 = getelementptr inbounds float, float* %arr, i64 %idxprom1
  %1 = load float, float* %arrayidx2, align 4
  %conv3 = fptoui float %1 to i64
  %vecinit4 = insertelement <2 x i64> %vecinit, i64 %conv3, i32 1
  ret <2 x i64> %vecinit4
; P9BE-LABEL: fromDiffMemVarDConvftoull
; P9LE-LABEL: fromDiffMemVarDConvftoull
; P8BE-LABEL: fromDiffMemVarDConvftoull
; P8LE-LABEL: fromDiffMemVarDConvftoull
; P9BE: sldi
; P9BE: lfsux
; P9BE: lfs
; P9BE: xxmrghd
; P9BE-NEXT: xvcvdpuxds v2
; P9BE-NEXT: blr
; P9LE: sldi
; P9LE: lfsux
; P9LE: lfs
; P9LE: xxmrghd
; P9LE-NEXT: xvcvdpuxds v2
; P9LE-NEXT: blr
; P8BE: sldi
; P8BE: lfsux
; P8BE: lfsx
; P8BE: xxmrghd
; P8BE-NEXT: xvcvdpuxds v2
; P8BE-NEXT: blr
; P8LE: sldi
; P8LE: lfsux
; P8LE: lfsx
; P8LE: xxmrghd
; P8LE-NEXT: xvcvdpuxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @spltRegValConvftoull(float %val) {
entry:
  %conv = fptoui float %val to i64
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %conv, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; P9BE-LABEL: spltRegValConvftoull
; P9LE-LABEL: spltRegValConvftoull
; P8BE-LABEL: spltRegValConvftoull
; P8LE-LABEL: spltRegValConvftoull
; P9BE: xscvdpuxds
; P9BE-NEXT: xxspltd v2
; P9BE-NEXT: blr
; P9LE: xscvdpuxds
; P9LE-NEXT: xxspltd v2
; P9LE-NEXT: blr
; P8BE: xscvdpuxds
; P8BE-NEXT: xxspltd v2
; P8BE-NEXT: blr
; P8LE: xscvdpuxds
; P8LE-NEXT: xxspltd v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @spltMemValConvftoull(float* nocapture readonly %ptr) {
entry:
  %0 = load float, float* %ptr, align 4
  %conv = fptoui float %0 to i64
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %conv, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; P9BE-LABEL: spltMemValConvftoull
; P9LE-LABEL: spltMemValConvftoull
; P8BE-LABEL: spltMemValConvftoull
; P8LE-LABEL: spltMemValConvftoull
; P9BE: lfs
; P9BE-NEXT: xscvdpuxds
; P9BE-NEXT: xxspltd v2
; P9BE-NEXT: blr
; P9LE: lfs
; P9LE-NEXT: xscvdpuxds
; P9LE-NEXT: xxspltd v2
; P9LE-NEXT: blr
; P8BE: lfsx
; P8BE-NEXT: xscvdpuxds
; P8BE-NEXT: xxspltd v2
; P8BE-NEXT: blr
; P8LE: lfsx
; P8LE-NEXT: xscvdpuxds
; P8LE-NEXT: xxspltd v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @spltCnstConvdtoull() {
entry:
  ret <2 x i64> <i64 4, i64 4>
; P9BE-LABEL: spltCnstConvdtoull
; P9LE-LABEL: spltCnstConvdtoull
; P8BE-LABEL: spltCnstConvdtoull
; P8LE-LABEL: spltCnstConvdtoull
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvd2x
; P8BE: blr
; P8LE: lxvd2x
; P8LE: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @fromRegsConvdtoull(double %a, double %b) {
entry:
  %conv = fptoui double %a to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %conv1 = fptoui double %b to i64
  %vecinit2 = insertelement <2 x i64> %vecinit, i64 %conv1, i32 1
  ret <2 x i64> %vecinit2
; P9BE-LABEL: fromRegsConvdtoull
; P9LE-LABEL: fromRegsConvdtoull
; P8BE-LABEL: fromRegsConvdtoull
; P8LE-LABEL: fromRegsConvdtoull
; P9BE: xxmrghd
; P9BE-NEXT: xvcvdpuxds
; P9BE-NEXT: blr
; P9LE: xxmrghd
; P9LE-NEXT: xvcvdpuxds
; P9LE-NEXT: blr
; P8BE: xxmrghd
; P8BE-NEXT: xvcvdpuxds
; P8BE-NEXT: blr
; P8LE: xxmrghd
; P8LE-NEXT: xvcvdpuxds
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @fromDiffConstsConvdtoull() {
entry:
  ret <2 x i64> <i64 24, i64 234>
; P9BE-LABEL: fromDiffConstsConvdtoull
; P9LE-LABEL: fromDiffConstsConvdtoull
; P8BE-LABEL: fromDiffConstsConvdtoull
; P8LE-LABEL: fromDiffConstsConvdtoull
; P9BE: lxv
; P9BE: blr
; P9LE: lxv
; P9LE: blr
; P8BE: lxvd2x
; P8BE: blr
; P8LE: lxvd2x
; P8LE: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemConsAConvdtoull(double* nocapture readonly %ptr) {
entry:
  %0 = bitcast double* %ptr to <2 x double>*
  %1 = load <2 x double>, <2 x double>* %0, align 8
  %2 = fptoui <2 x double> %1 to <2 x i64>
  ret <2 x i64> %2
; P9BE-LABEL: fromDiffMemConsAConvdtoull
; P9LE-LABEL: fromDiffMemConsAConvdtoull
; P8BE-LABEL: fromDiffMemConsAConvdtoull
; P8LE-LABEL: fromDiffMemConsAConvdtoull
; P9BE: lxv
; P9BE-NEXT: xvcvdpuxds v2
; P9BE-NEXT: blr
; P9LE: lxv
; P9LE-NEXT: xvcvdpuxds v2
; P9LE-NEXT: blr
; P8BE: lxvd2x
; P8BE-NEXT: xvcvdpuxds v2
; P8BE-NEXT: blr
; P8LE: lxvd2x
; P8LE: xxswapd
; P8LE-NEXT: xvcvdpuxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemConsDConvdtoull(double* nocapture readonly %ptr) {
entry:
  %arrayidx = getelementptr inbounds double, double* %ptr, i64 3
  %0 = load double, double* %arrayidx, align 8
  %conv = fptoui double %0 to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %arrayidx1 = getelementptr inbounds double, double* %ptr, i64 2
  %1 = load double, double* %arrayidx1, align 8
  %conv2 = fptoui double %1 to i64
  %vecinit3 = insertelement <2 x i64> %vecinit, i64 %conv2, i32 1
  ret <2 x i64> %vecinit3
; P9BE-LABEL: fromDiffMemConsDConvdtoull
; P9LE-LABEL: fromDiffMemConsDConvdtoull
; P8BE-LABEL: fromDiffMemConsDConvdtoull
; P8LE-LABEL: fromDiffMemConsDConvdtoull
; P9BE: lxv
; P9BE-NEXT: xxswapd
; P9BE-NEXT: xvcvdpuxds v2
; P9BE-NEXT: blr
; P9LE: lxv
; P9LE-NEXT: xxswapd
; P9LE-NEXT: xvcvdpuxds v2
; P9LE-NEXT: blr
; P8BE: lxvd2x
; P8BE-NEXT: xxswapd
; P8BE-NEXT: xvcvdpuxds v2
; P8BE-NEXT: blr
; P8LE: lxvd2x
; P8LE-NEXT: xvcvdpuxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemVarAConvdtoull(double* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds double, double* %arr, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
  %conv = fptoui double %0 to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %add = add nsw i32 %elem, 1
  %idxprom1 = sext i32 %add to i64
  %arrayidx2 = getelementptr inbounds double, double* %arr, i64 %idxprom1
  %1 = load double, double* %arrayidx2, align 8
  %conv3 = fptoui double %1 to i64
  %vecinit4 = insertelement <2 x i64> %vecinit, i64 %conv3, i32 1
  ret <2 x i64> %vecinit4
; P9BE-LABEL: fromDiffMemVarAConvdtoull
; P9LE-LABEL: fromDiffMemVarAConvdtoull
; P8BE-LABEL: fromDiffMemVarAConvdtoull
; P8LE-LABEL: fromDiffMemVarAConvdtoull
; P9BE: sldi
; P9BE: lxvx
; P9BE-NEXT: xvcvdpuxds v2
; P9BE-NEXT: blr
; P9LE: sldi
; P9LE: lxvx
; P9LE-NEXT: xvcvdpuxds v2
; P9LE-NEXT: blr
; P8BE: sldi
; P8BE: lxvd2x
; P8BE-NEXT: xvcvdpuxds v2
; P8BE-NEXT: blr
; P8LE: sldi
; P8LE: lxvd2x
; P8LE-NEXT: xxswapd
; P8LE-NEXT: xvcvdpuxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @fromDiffMemVarDConvdtoull(double* nocapture readonly %arr, i32 signext %elem) {
entry:
  %idxprom = sext i32 %elem to i64
  %arrayidx = getelementptr inbounds double, double* %arr, i64 %idxprom
  %0 = load double, double* %arrayidx, align 8
  %conv = fptoui double %0 to i64
  %vecinit = insertelement <2 x i64> undef, i64 %conv, i32 0
  %sub = add nsw i32 %elem, -1
  %idxprom1 = sext i32 %sub to i64
  %arrayidx2 = getelementptr inbounds double, double* %arr, i64 %idxprom1
  %1 = load double, double* %arrayidx2, align 8
  %conv3 = fptoui double %1 to i64
  %vecinit4 = insertelement <2 x i64> %vecinit, i64 %conv3, i32 1
  ret <2 x i64> %vecinit4
; P9BE-LABEL: fromDiffMemVarDConvdtoull
; P9LE-LABEL: fromDiffMemVarDConvdtoull
; P8BE-LABEL: fromDiffMemVarDConvdtoull
; P8LE-LABEL: fromDiffMemVarDConvdtoull
; P9BE: sldi
; P9BE: lxv
; P9BE-NEXT: xxswapd
; P9BE-NEXT: xvcvdpuxds v2
; P9BE-NEXT: blr
; P9LE: sldi
; P9LE: lxv
; P9LE-NEXT: xxswapd
; P9LE-NEXT: xvcvdpuxds v2
; P9LE-NEXT: blr
; P8BE: sldi
; P8BE: lxvd2x
; P8BE-NEXT: xxswapd
; P8BE-NEXT: xvcvdpuxds v2
; P8BE-NEXT: blr
; P8LE: sldi
; P8LE: lxvd2x
; P8LE-NEXT: xvcvdpuxds v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readnone
define <2 x i64> @spltRegValConvdtoull(double %val) {
entry:
  %conv = fptoui double %val to i64
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %conv, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; P9BE-LABEL: spltRegValConvdtoull
; P9LE-LABEL: spltRegValConvdtoull
; P8BE-LABEL: spltRegValConvdtoull
; P8LE-LABEL: spltRegValConvdtoull
; P9BE: xscvdpuxds
; P9BE-NEXT: xxspltd v2
; P9BE-NEXT: blr
; P9LE: xscvdpuxds
; P9LE-NEXT: xxspltd v2
; P9LE-NEXT: blr
; P8BE: xscvdpuxds
; P8BE-NEXT: xxspltd v2
; P8BE-NEXT: blr
; P8LE: xscvdpuxds
; P8LE-NEXT: xxspltd v2
; P8LE-NEXT: blr
}

; Function Attrs: norecurse nounwind readonly
define <2 x i64> @spltMemValConvdtoull(double* nocapture readonly %ptr) {
entry:
  %0 = load double, double* %ptr, align 8
  %conv = fptoui double %0 to i64
  %splat.splatinsert = insertelement <2 x i64> undef, i64 %conv, i32 0
  %splat.splat = shufflevector <2 x i64> %splat.splatinsert, <2 x i64> undef, <2 x i32> zeroinitializer
  ret <2 x i64> %splat.splat
; P9BE-LABEL: spltMemValConvdtoull
; P9LE-LABEL: spltMemValConvdtoull
; P8BE-LABEL: spltMemValConvdtoull
; P8LE-LABEL: spltMemValConvdtoull
; P9BE: lxvdsx
; P9BE-NEXT: xvcvdpuxds
; P9BE-NEXT: blr
; P9LE: lxvdsx
; P9LE-NEXT: xvcvdpuxds
; P9LE-NEXT: blr
; P8BE: lxvdsx
; P8BE-NEXT: xvcvdpuxds
; P8BE-NEXT: blr
; P8LE: lxvdsx
; P8LE-NEXT: xvcvdpuxds
; P8LE-NEXT: blr
}
