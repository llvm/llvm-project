; RUN: llvm-mc -triple aarch64-apple-ios -mattr=+lse %s | FileCheck %s

    ; CHECK: ldaddab w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldaddah w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: ldadda w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldadda x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    ldaddab w3, wzr, [x5]
    ldaddah w5, wzr, [x0]
    ldadda w7, wzr, [x5]
    ldadda x9, xzr, [sp]

    ; CHECK: ldaddalb w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldaddalh w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: ldaddal w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldaddal x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    ldaddalb w3, wzr, [x5]
    ldaddalh w5, wzr, [x0]
    ldaddal w7, wzr, [x5]
    ldaddal x9, xzr, [sp]

    ; CHECK: ldclrab w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldclrah w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: ldclra w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldclra x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    ldclrab w3, wzr, [x5]
    ldclrah w5, wzr, [x0]
    ldclra w7, wzr, [x5]
    ldclra x9, xzr, [sp]

    ; CHECK: ldclralb w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldclralh w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: ldclral w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldclral x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    ldclralb w3, wzr, [x5]
    ldclralh w5, wzr, [x0]
    ldclral w7, wzr, [x5]
    ldclral x9, xzr, [sp]

    ; CHECK: ldeorab w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldeorah w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: ldeora w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldeora x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    ldeorab w3, wzr, [x5]
    ldeorah w5, wzr, [x0]
    ldeora w7, wzr, [x5]
    ldeora x9, xzr, [sp]

    ; CHECK: ldeoralb w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldeoralh w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: ldeoral w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldeoral x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    ldeoralb w3, wzr, [x5]
    ldeoralh w5, wzr, [x0]
    ldeoral w7, wzr, [x5]
    ldeoral x9, xzr, [sp]

    ; CHECK: ldsetab w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldsetah w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: ldseta w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldseta x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    ldsetab w3, wzr, [x5]
    ldsetah w5, wzr, [x0]
    ldseta w7, wzr, [x5]
    ldseta x9, xzr, [sp]

    ; CHECK: ldsetalb w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldsetalh w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: ldsetal w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldsetal x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    ldsetalb w3, wzr, [x5]
    ldsetalh w5, wzr, [x0]
    ldsetal w7, wzr, [x5]
    ldsetal x9, xzr, [sp]

    ; CHECK: ldsmaxab w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldsmaxah w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: ldsmaxa w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldsmaxa x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    ldsmaxab w3, wzr, [x5]
    ldsmaxah w5, wzr, [x0]
    ldsmaxa w7, wzr, [x5]
    ldsmaxa x9, xzr, [sp]

    ; CHECK: ldsmaxalb w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldsmaxalh w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: ldsmaxal w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldsmaxal x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    ldsmaxalb w3, wzr, [x5]
    ldsmaxalh w5, wzr, [x0]
    ldsmaxal w7, wzr, [x5]
    ldsmaxal x9, xzr, [sp]

    ; CHECK: ldsminab w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldsminah w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: ldsmina w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldsmina x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    ldsminab w3, wzr, [x5]
    ldsminah w5, wzr, [x0]
    ldsmina w7, wzr, [x5]
    ldsmina x9, xzr, [sp]

    ; CHECK: ldsminalb w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldsminalh w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: ldsminal w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldsminal x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    ldsminalb w3, wzr, [x5]
    ldsminalh w5, wzr, [x0]
    ldsminal w7, wzr, [x5]
    ldsminal x9, xzr, [sp]

    ; CHECK: ldumaxab w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldumaxah w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: ldumaxa w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldumaxa x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    ldumaxab w3, wzr, [x5]
    ldumaxah w5, wzr, [x0]
    ldumaxa w7, wzr, [x5]
    ldumaxa x9, xzr, [sp]

    ; CHECK: ldumaxalb w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldumaxalh w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: ldumaxal w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldumaxal x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    ldumaxalb w3, wzr, [x5]
    ldumaxalh w5, wzr, [x0]
    ldumaxal w7, wzr, [x5]
    ldumaxal x9, xzr, [sp]

    ; CHECK: lduminab w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: lduminah w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: ldumina w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: ldumina x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    lduminab w3, wzr, [x5]
    lduminah w5, wzr, [x0]
    ldumina w7, wzr, [x5]
    ldumina x9, xzr, [sp]

    ; CHECK: lduminalb w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: lduminalh w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: lduminal w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: lduminal x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    lduminalb w3, wzr, [x5]
    lduminalh w5, wzr, [x0]
    lduminal w7, wzr, [x5]
    lduminal x9, xzr, [sp]

    ; CHECK: swpab w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: swpah w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: swpa w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: swpa x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    swpab w3, wzr, [x5]
    swpah w5, wzr, [x0]
    swpa w7, wzr, [x5]
    swpa x9, xzr, [sp]

    ; CHECK: swpalb w3, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: swpalh w5, wzr, [x0] ; acquire semantics dropped since destination is zero
    ; CHECK: swpal w7, wzr, [x5] ; acquire semantics dropped since destination is zero
    ; CHECK: swpal x9, xzr, [sp] ; acquire semantics dropped since destination is zero
    swpalb w3, wzr, [x5]
    swpalh w5, wzr, [x0]
    swpal w7, wzr, [x5]
    swpal x9, xzr, [sp]

    ; CHECK: ldaddal xzr, x3, [x0]{{$}}
    ; CHECK: ldeora wzr, w7, [x5]{{$}}
    ; CHECK: ldsminb w5, w9, [sp]{{$}}
    ldaddal xzr, x3, [x0]
    ldeora wzr, w7, [x5]
    ldsminb w5, w9, [sp]

    ; CAS instructions aren't affected.

    ; CHECK: casab w3, wzr, [x5]{{$}}
    ; CHECK: casah w5, wzr, [x0]{{$}}
    ; CHECK: casa w7, wzr, [x5]{{$}}
    ; CHECK: casa x9, xzr, [sp]{{$}}
    casab w3, wzr, [x5]
    casah w5, wzr, [x0]
    casa w7, wzr, [x5]
    casa x9, xzr, [sp]

    ; CHECK: casalb w3, wzr, [x5]{{$}}
    ; CHECK: casalh w5, wzr, [x0]{{$}}
    ; CHECK: casal w7, wzr, [x5]{{$}}
    ; CHECK: casal x9, xzr, [sp]{{$}}
    casalb w3, wzr, [x5]
    casalh w5, wzr, [x0]
    casal w7, wzr, [x5]
    casal x9, xzr, [sp]
