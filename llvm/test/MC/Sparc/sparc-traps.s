! RUN: llvm-mc %s -triple=sparc -show-encoding | FileCheck %s

        ! CHECK: ta %i5          ! encoding: [0x91,0xd0,0x00,0x1d]
        ! CHECK: ta 82           ! encoding: [0x91,0xd0,0x20,0x52]
        ! CHECK: ta %g1 + %i2    ! encoding: [0x91,0xd0,0x40,0x1a]
        ! CHECK: ta %i5 + 41     ! encoding: [0x91,0xd7,0x60,0x29]
        ta %i5
        ta 82
        ta %g1 + %i2
        ta %i5 + 41

        ! CHECK: tn %i5          ! encoding: [0x81,0xd0,0x00,0x1d]
        ! CHECK: tn 82           ! encoding: [0x81,0xd0,0x20,0x52]
        ! CHECK: tn %g1 + %i2    ! encoding: [0x81,0xd0,0x40,0x1a]
        ! CHECK: tn %i5 + 41     ! encoding: [0x81,0xd7,0x60,0x29]
        tn %i5
        tn 82
        tn %g1 + %i2
        tn %i5 + 41

        ! CHECK: tne %i5         ! encoding: [0x93,0xd0,0x00,0x1d]
        !! tnz should be a synonym for tne
        ! CHECK: tne %i5         ! encoding: [0x93,0xd0,0x00,0x1d]
        ! CHECK: tne 82          ! encoding: [0x93,0xd0,0x20,0x52]
        ! CHECK: tne %g1 + %i2   ! encoding: [0x93,0xd0,0x40,0x1a]
        ! CHECK: tne %i5 + 41    ! encoding: [0x93,0xd7,0x60,0x29]
        tne %i5
        tnz %i5
        tne 82
        tne %g1 + %i2
        tne %i5 + 41

        ! CHECK: te %i5          ! encoding: [0x83,0xd0,0x00,0x1d]
        !! tz should be a synonym for te
        ! CHECK: te %i5          ! encoding: [0x83,0xd0,0x00,0x1d]
        ! CHECK: te 82           ! encoding: [0x83,0xd0,0x20,0x52]
        ! CHECK: te %g1 + %i2    ! encoding: [0x83,0xd0,0x40,0x1a]
        ! CHECK: te %i5 + 41     ! encoding: [0x83,0xd7,0x60,0x29]
        te %i5
        tz %i5
        te 82
        te %g1 + %i2
        te %i5 + 41

        ! CHECK: tg %i5          ! encoding: [0x95,0xd0,0x00,0x1d]
        ! CHECK: tg 82           ! encoding: [0x95,0xd0,0x20,0x52]
        ! CHECK: tg %g1 + %i2    ! encoding: [0x95,0xd0,0x40,0x1a]
        ! CHECK: tg %i5 + 41     ! encoding: [0x95,0xd7,0x60,0x29]
        tg %i5
        tg 82
        tg %g1 + %i2
        tg %i5 + 41

        ! CHECK: tle %i5         ! encoding: [0x85,0xd0,0x00,0x1d]
        ! CHECK: tle 82          ! encoding: [0x85,0xd0,0x20,0x52]
        ! CHECK: tle %g1 + %i2   ! encoding: [0x85,0xd0,0x40,0x1a]
        ! CHECK: tle %i5 + 41    ! encoding: [0x85,0xd7,0x60,0x29]
        tle %i5
        tle 82
        tle %g1 + %i2
        tle %i5 + 41

        ! CHECK: tge %i5         ! encoding: [0x97,0xd0,0x00,0x1d]
        ! CHECK: tge 82          ! encoding: [0x97,0xd0,0x20,0x52]
        ! CHECK: tge %g1 + %i2   ! encoding: [0x97,0xd0,0x40,0x1a]
        ! CHECK: tge %i5 + 41    ! encoding: [0x97,0xd7,0x60,0x29]
        tge %i5
        tge 82
        tge %g1 + %i2
        tge %i5 + 41

        ! CHECK: tl %i5          ! encoding: [0x87,0xd0,0x00,0x1d]
        ! CHECK: tl 82           ! encoding: [0x87,0xd0,0x20,0x52]
        ! CHECK: tl %g1 + %i2    ! encoding: [0x87,0xd0,0x40,0x1a]
        ! CHECK: tl %i5 + 41     ! encoding: [0x87,0xd7,0x60,0x29]
        tl %i5
        tl 82
        tl %g1 + %i2
        tl %i5 + 41

        ! CHECK: tgu %i5         ! encoding: [0x99,0xd0,0x00,0x1d]
        ! CHECK: tgu 82          ! encoding: [0x99,0xd0,0x20,0x52]
        ! CHECK: tgu %g1 + %i2   ! encoding: [0x99,0xd0,0x40,0x1a]
        ! CHECK: tgu %i5 + 41    ! encoding: [0x99,0xd7,0x60,0x29]
        tgu %i5
        tgu 82
        tgu %g1 + %i2
        tgu %i5 + 41

        ! CHECK: tleu %i5        ! encoding: [0x89,0xd0,0x00,0x1d]
        ! CHECK: tleu 82         ! encoding: [0x89,0xd0,0x20,0x52]
        ! CHECK: tleu %g1 + %i2  ! encoding: [0x89,0xd0,0x40,0x1a]
        ! CHECK: tleu %i5 + 41   ! encoding: [0x89,0xd7,0x60,0x29]
        tleu %i5
        tleu 82
        tleu %g1 + %i2
        tleu %i5 + 41

        ! CHECK: tcc %i5         ! encoding: [0x9b,0xd0,0x00,0x1d]
        ! CHECK: tcc 82          ! encoding: [0x9b,0xd0,0x20,0x52]
        ! CHECK: tcc %g1 + %i2   ! encoding: [0x9b,0xd0,0x40,0x1a]
        ! CHECK: tcc %i5 + 41    ! encoding: [0x9b,0xd7,0x60,0x29]
        tcc %i5
        tcc 82
        tcc %g1 + %i2
        tcc %i5 + 41

        ! CHECK: tcs %i5         ! encoding: [0x8b,0xd0,0x00,0x1d]
        ! CHECK: tcs 82          ! encoding: [0x8b,0xd0,0x20,0x52]
        ! CHECK: tcs %g1 + %i2   ! encoding: [0x8b,0xd0,0x40,0x1a]
        ! CHECK: tcs %i5 + 41    ! encoding: [0x8b,0xd7,0x60,0x29]
        tcs %i5
        tcs 82
        tcs %g1 + %i2
        tcs %i5 + 41

        ! CHECK: tpos %i5        ! encoding: [0x9d,0xd0,0x00,0x1d]
        ! CHECK: tpos 82         ! encoding: [0x9d,0xd0,0x20,0x52]
        ! CHECK: tpos %g1 + %i2  ! encoding: [0x9d,0xd0,0x40,0x1a]
        ! CHECK: tpos %i5 + 41   ! encoding: [0x9d,0xd7,0x60,0x29]
        tpos %i5
        tpos 82
        tpos %g1 + %i2
        tpos %i5 + 41

        ! CHECK: tneg %i5        ! encoding: [0x8d,0xd0,0x00,0x1d]
        ! CHECK: tneg 82         ! encoding: [0x8d,0xd0,0x20,0x52]
        ! CHECK: tneg %g1 + %i2  ! encoding: [0x8d,0xd0,0x40,0x1a]
        ! CHECK: tneg %i5 + 41   ! encoding: [0x8d,0xd7,0x60,0x29]
        tneg %i5
        tneg 82
        tneg %g1 + %i2
        tneg %i5 + 41

        ! CHECK: tvc %i5         ! encoding: [0x9f,0xd0,0x00,0x1d]
        ! CHECK: tvc 82          ! encoding: [0x9f,0xd0,0x20,0x52]
        ! CHECK: tvc %g1 + %i2   ! encoding: [0x9f,0xd0,0x40,0x1a]
        ! CHECK: tvc %i5 + 41    ! encoding: [0x9f,0xd7,0x60,0x29]
        tvc %i5
        tvc 82
        tvc %g1 + %i2
        tvc %i5 + 41

        ! CHECK: tvs %i5         ! encoding: [0x8f,0xd0,0x00,0x1d]
        ! CHECK: tvs 82          ! encoding: [0x8f,0xd0,0x20,0x52]
        ! CHECK: tvs %g1 + %i2   ! encoding: [0x8f,0xd0,0x40,0x1a]
        ! CHECK: tvs %i5 + 41    ! encoding: [0x8f,0xd7,0x60,0x29]
        tvs %i5
        tvs 82
        tvs %g1 + %i2
        tvs %i5 + 41
