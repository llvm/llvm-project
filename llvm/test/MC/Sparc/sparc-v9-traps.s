! RUN: llvm-mc %s -arch=sparcv9 -show-encoding | FileCheck %s

        ! CHECK: ta %icc, %i5           ! encoding: [0x91,0xd0,0x00,0x1d]
        ! CHECK: ta %icc, 82            ! encoding: [0x91,0xd0,0x20,0x52]
        ! CHECK: ta %icc, %g1 + %i2     ! encoding: [0x91,0xd0,0x40,0x1a]
        ! CHECK: ta %icc, %i5 + 41      ! encoding: [0x91,0xd7,0x60,0x29]
        ta %icc, %i5
        ta %icc, 82
        ta %icc, %g1 + %i2
        ta %icc, %i5 + 41

        ! CHECK: tn %icc, %i5           ! encoding: [0x81,0xd0,0x00,0x1d]
        ! CHECK: tn %icc, 82            ! encoding: [0x81,0xd0,0x20,0x52]
        ! CHECK: tn %icc, %g1 + %i2     ! encoding: [0x81,0xd0,0x40,0x1a]
        ! CHECK: tn %icc, %i5 + 41      ! encoding: [0x81,0xd7,0x60,0x29]
        tn %icc, %i5
        tn %icc, 82
        tn %icc, %g1 + %i2
        tn %icc, %i5 + 41

        ! CHECK: tne %icc, %i5          ! encoding: [0x93,0xd0,0x00,0x1d]
        !! tnz should be a synonym for tne
        ! CHECK: tne %icc, %i5          ! encoding: [0x93,0xd0,0x00,0x1d]
        ! CHECK: tne %icc, 82           ! encoding: [0x93,0xd0,0x20,0x52]
        ! CHECK: tne %icc, %g1 + %i2    ! encoding: [0x93,0xd0,0x40,0x1a]
        ! CHECK: tne %icc, %i5 + 41     ! encoding: [0x93,0xd7,0x60,0x29]
        tne %icc, %i5
        tnz %icc, %i5
        tne %icc, 82
        tne %icc, %g1 + %i2
        tne %icc, %i5 + 41

        ! CHECK: te %icc, %i5           ! encoding: [0x83,0xd0,0x00,0x1d]
        !! tz should be a synonym for te
        ! CHECK: te %icc, %i5           ! encoding: [0x83,0xd0,0x00,0x1d]
        ! CHECK: te %icc, 82            ! encoding: [0x83,0xd0,0x20,0x52]
        ! CHECK: te %icc, %g1 + %i2     ! encoding: [0x83,0xd0,0x40,0x1a]
        ! CHECK: te %icc, %i5 + 41      ! encoding: [0x83,0xd7,0x60,0x29]
        te %icc, %i5
        tz %icc, %i5
        te %icc, 82
        te %icc, %g1 + %i2
        te %icc, %i5 + 41

        ! CHECK: tg %icc, %i5           ! encoding: [0x95,0xd0,0x00,0x1d]
        ! CHECK: tg %icc, 82            ! encoding: [0x95,0xd0,0x20,0x52]
        ! CHECK: tg %icc, %g1 + %i2     ! encoding: [0x95,0xd0,0x40,0x1a]
        ! CHECK: tg %icc, %i5 + 41      ! encoding: [0x95,0xd7,0x60,0x29]
        tg %icc, %i5
        tg %icc, 82
        tg %icc, %g1 + %i2
        tg %icc, %i5 + 41

        ! CHECK: tle %icc, %i5          ! encoding: [0x85,0xd0,0x00,0x1d]
        ! CHECK: tle %icc, 82           ! encoding: [0x85,0xd0,0x20,0x52]
        ! CHECK: tle %icc, %g1 + %i2    ! encoding: [0x85,0xd0,0x40,0x1a]
        ! CHECK: tle %icc, %i5 + 41     ! encoding: [0x85,0xd7,0x60,0x29]
        tle %icc, %i5
        tle %icc, 82
        tle %icc, %g1 + %i2
        tle %icc, %i5 + 41

        ! CHECK: tge %icc, %i5          ! encoding: [0x97,0xd0,0x00,0x1d]
        ! CHECK: tge %icc, 82           ! encoding: [0x97,0xd0,0x20,0x52]
        ! CHECK: tge %icc, %g1 + %i2    ! encoding: [0x97,0xd0,0x40,0x1a]
        ! CHECK: tge %icc, %i5 + 41     ! encoding: [0x97,0xd7,0x60,0x29]
        tge %icc, %i5
        tge %icc, 82
        tge %icc, %g1 + %i2
        tge %icc, %i5 + 41

        ! CHECK: tl %icc, %i5           ! encoding: [0x87,0xd0,0x00,0x1d]
        ! CHECK: tl %icc, 82            ! encoding: [0x87,0xd0,0x20,0x52]
        ! CHECK: tl %icc, %g1 + %i2     ! encoding: [0x87,0xd0,0x40,0x1a]
        ! CHECK: tl %icc, %i5 + 41      ! encoding: [0x87,0xd7,0x60,0x29]
        tl %icc, %i5
        tl %icc, 82
        tl %icc, %g1 + %i2
        tl %icc, %i5 + 41

        ! CHECK: tgu %icc, %i5          ! encoding: [0x99,0xd0,0x00,0x1d]
        ! CHECK: tgu %icc, 82           ! encoding: [0x99,0xd0,0x20,0x52]
        ! CHECK: tgu %icc, %g1 + %i2    ! encoding: [0x99,0xd0,0x40,0x1a]
        ! CHECK: tgu %icc, %i5 + 41     ! encoding: [0x99,0xd7,0x60,0x29]
        tgu %icc, %i5
        tgu %icc, 82
        tgu %icc, %g1 + %i2
        tgu %icc, %i5 + 41

        ! CHECK: tleu %icc, %i5         ! encoding: [0x89,0xd0,0x00,0x1d]
        ! CHECK: tleu %icc, 82          ! encoding: [0x89,0xd0,0x20,0x52]
        ! CHECK: tleu %icc, %g1 + %i2   ! encoding: [0x89,0xd0,0x40,0x1a]
        ! CHECK: tleu %icc, %i5 + 41    ! encoding: [0x89,0xd7,0x60,0x29]
        tleu %icc, %i5
        tleu %icc, 82
        tleu %icc, %g1 + %i2
        tleu %icc, %i5 + 41

        ! CHECK: tcc %icc, %i5          ! encoding: [0x9b,0xd0,0x00,0x1d]
        ! CHECK: tcc %icc, 82           ! encoding: [0x9b,0xd0,0x20,0x52]
        ! CHECK: tcc %icc, %g1 + %i2    ! encoding: [0x9b,0xd0,0x40,0x1a]
        ! CHECK: tcc %icc, %i5 + 41     ! encoding: [0x9b,0xd7,0x60,0x29]
        tcc %icc, %i5
        tcc %icc, 82
        tcc %icc, %g1 + %i2
        tcc %icc, %i5 + 41

        ! CHECK: tcs %icc, %i5          ! encoding: [0x8b,0xd0,0x00,0x1d]
        ! CHECK: tcs %icc, 82           ! encoding: [0x8b,0xd0,0x20,0x52]
        ! CHECK: tcs %icc, %g1 + %i2    ! encoding: [0x8b,0xd0,0x40,0x1a]
        ! CHECK: tcs %icc, %i5 + 41     ! encoding: [0x8b,0xd7,0x60,0x29]
        tcs %icc, %i5
        tcs %icc, 82
        tcs %icc, %g1 + %i2
        tcs %icc, %i5 + 41

        ! CHECK: tpos %icc, %i5         ! encoding: [0x9d,0xd0,0x00,0x1d]
        ! CHECK: tpos %icc, 82          ! encoding: [0x9d,0xd0,0x20,0x52]
        ! CHECK: tpos %icc, %g1 + %i2   ! encoding: [0x9d,0xd0,0x40,0x1a]
        ! CHECK: tpos %icc, %i5 + 41    ! encoding: [0x9d,0xd7,0x60,0x29]
        tpos %icc, %i5
        tpos %icc, 82
        tpos %icc, %g1 + %i2
        tpos %icc, %i5 + 41

        ! CHECK: tneg %icc, %i5         ! encoding: [0x8d,0xd0,0x00,0x1d]
        ! CHECK: tneg %icc, 82          ! encoding: [0x8d,0xd0,0x20,0x52]
        ! CHECK: tneg %icc, %g1 + %i2   ! encoding: [0x8d,0xd0,0x40,0x1a]
        ! CHECK: tneg %icc, %i5 + 41    ! encoding: [0x8d,0xd7,0x60,0x29]
        tneg %icc, %i5
        tneg %icc, 82
        tneg %icc, %g1 + %i2
        tneg %icc, %i5 + 41

        ! CHECK: tvc %icc, %i5          ! encoding: [0x9f,0xd0,0x00,0x1d]
        ! CHECK: tvc %icc, 82           ! encoding: [0x9f,0xd0,0x20,0x52]
        ! CHECK: tvc %icc, %g1 + %i2    ! encoding: [0x9f,0xd0,0x40,0x1a]
        ! CHECK: tvc %icc, %i5 + 41     ! encoding: [0x9f,0xd7,0x60,0x29]
        tvc %icc, %i5
        tvc %icc, 82
        tvc %icc, %g1 + %i2
        tvc %icc, %i5 + 41

        ! CHECK: tvs %icc, %i5          ! encoding: [0x8f,0xd0,0x00,0x1d]
        ! CHECK: tvs %icc, 82           ! encoding: [0x8f,0xd0,0x20,0x52]
        ! CHECK: tvs %icc, %g1 + %i2    ! encoding: [0x8f,0xd0,0x40,0x1a]
        ! CHECK: tvs %icc, %i5 + 41     ! encoding: [0x8f,0xd7,0x60,0x29]
        tvs %icc, %i5
        tvs %icc, 82
        tvs %icc, %g1 + %i2
        tvs %icc, %i5 + 41


        ! CHECK: ta %xcc, %i5           ! encoding: [0x91,0xd0,0x10,0x1d]
        ! CHECK: ta %xcc, 82            ! encoding: [0x91,0xd0,0x30,0x52]
        ! CHECK: ta %xcc, %g1 + %i2     ! encoding: [0x91,0xd0,0x50,0x1a]
        ! CHECK: ta %xcc, %i5 + 41      ! encoding: [0x91,0xd7,0x70,0x29]
        ta %xcc, %i5
        ta %xcc, 82
        ta %xcc, %g1 + %i2
        ta %xcc, %i5 + 41

        ! CHECK: tn %xcc, %i5           ! encoding: [0x81,0xd0,0x10,0x1d]
        ! CHECK: tn %xcc, 82            ! encoding: [0x81,0xd0,0x30,0x52]
        ! CHECK: tn %xcc, %g1 + %i2     ! encoding: [0x81,0xd0,0x50,0x1a]
        ! CHECK: tn %xcc, %i5 + 41      ! encoding: [0x81,0xd7,0x70,0x29]
        tn %xcc, %i5
        tn %xcc, 82
        tn %xcc, %g1 + %i2
        tn %xcc, %i5 + 41

        ! CHECK: tne %xcc, %i5          ! encoding: [0x93,0xd0,0x10,0x1d]
        !! tnz should be a synonym for tne
        ! CHECK: tne %xcc, %i5          ! encoding: [0x93,0xd0,0x10,0x1d]
        ! CHECK: tne %xcc, 82           ! encoding: [0x93,0xd0,0x30,0x52]
        ! CHECK: tne %xcc, %g1 + %i2    ! encoding: [0x93,0xd0,0x50,0x1a]
        ! CHECK: tne %xcc, %i5 + 41     ! encoding: [0x93,0xd7,0x70,0x29]
        tne %xcc, %i5
        tnz %xcc, %i5
        tne %xcc, 82
        tne %xcc, %g1 + %i2
        tne %xcc, %i5 + 41

        ! CHECK: te %xcc, %i5           ! encoding: [0x83,0xd0,0x10,0x1d]
        !! tz should be a synonym for te
        ! CHECK: te %xcc, %i5           ! encoding: [0x83,0xd0,0x10,0x1d]
        ! CHECK: te %xcc, 82            ! encoding: [0x83,0xd0,0x30,0x52]
        ! CHECK: te %xcc, %g1 + %i2     ! encoding: [0x83,0xd0,0x50,0x1a]
        ! CHECK: te %xcc, %i5 + 41      ! encoding: [0x83,0xd7,0x70,0x29]
        te %xcc, %i5
        tz %xcc, %i5
        te %xcc, 82
        te %xcc, %g1 + %i2
        te %xcc, %i5 + 41

        ! CHECK: tg %xcc, %i5           ! encoding: [0x95,0xd0,0x10,0x1d]
        ! CHECK: tg %xcc, 82            ! encoding: [0x95,0xd0,0x30,0x52]
        ! CHECK: tg %xcc, %g1 + %i2     ! encoding: [0x95,0xd0,0x50,0x1a]
        ! CHECK: tg %xcc, %i5 + 41      ! encoding: [0x95,0xd7,0x70,0x29]
        tg %xcc, %i5
        tg %xcc, 82
        tg %xcc, %g1 + %i2
        tg %xcc, %i5 + 41

        ! CHECK: tle %xcc, %i5          ! encoding: [0x85,0xd0,0x10,0x1d]
        ! CHECK: tle %xcc, 82           ! encoding: [0x85,0xd0,0x30,0x52]
        ! CHECK: tle %xcc, %g1 + %i2    ! encoding: [0x85,0xd0,0x50,0x1a]
        ! CHECK: tle %xcc, %i5 + 41     ! encoding: [0x85,0xd7,0x70,0x29]
        tle %xcc, %i5
        tle %xcc, 82
        tle %xcc, %g1 + %i2
        tle %xcc, %i5 + 41

        ! CHECK: tge %xcc, %i5          ! encoding: [0x97,0xd0,0x10,0x1d]
        ! CHECK: tge %xcc, 82           ! encoding: [0x97,0xd0,0x30,0x52]
        ! CHECK: tge %xcc, %g1 + %i2    ! encoding: [0x97,0xd0,0x50,0x1a]
        ! CHECK: tge %xcc, %i5 + 41     ! encoding: [0x97,0xd7,0x70,0x29]
        tge %xcc, %i5
        tge %xcc, 82
        tge %xcc, %g1 + %i2
        tge %xcc, %i5 + 41

        ! CHECK: tl %xcc, %i5           ! encoding: [0x87,0xd0,0x10,0x1d]
        ! CHECK: tl %xcc, 82            ! encoding: [0x87,0xd0,0x30,0x52]
        ! CHECK: tl %xcc, %g1 + %i2     ! encoding: [0x87,0xd0,0x50,0x1a]
        ! CHECK: tl %xcc, %i5 + 41      ! encoding: [0x87,0xd7,0x70,0x29]
        tl %xcc, %i5
        tl %xcc, 82
        tl %xcc, %g1 + %i2
        tl %xcc, %i5 + 41

        ! CHECK: tgu %xcc, %i5          ! encoding: [0x99,0xd0,0x10,0x1d]
        ! CHECK: tgu %xcc, 82           ! encoding: [0x99,0xd0,0x30,0x52]
        ! CHECK: tgu %xcc, %g1 + %i2    ! encoding: [0x99,0xd0,0x50,0x1a]
        ! CHECK: tgu %xcc, %i5 + 41     ! encoding: [0x99,0xd7,0x70,0x29]
        tgu %xcc, %i5
        tgu %xcc, 82
        tgu %xcc, %g1 + %i2
        tgu %xcc, %i5 + 41

        ! CHECK: tleu %xcc, %i5         ! encoding: [0x89,0xd0,0x10,0x1d]
        ! CHECK: tleu %xcc, 82          ! encoding: [0x89,0xd0,0x30,0x52]
        ! CHECK: tleu %xcc, %g1 + %i2   ! encoding: [0x89,0xd0,0x50,0x1a]
        ! CHECK: tleu %xcc, %i5 + 41    ! encoding: [0x89,0xd7,0x70,0x29]
        tleu %xcc, %i5
        tleu %xcc, 82
        tleu %xcc, %g1 + %i2
        tleu %xcc, %i5 + 41

        ! CHECK: tcc %xcc, %i5          ! encoding: [0x9b,0xd0,0x10,0x1d]
        ! CHECK: tcc %xcc, 82           ! encoding: [0x9b,0xd0,0x30,0x52]
        ! CHECK: tcc %xcc, %g1 + %i2    ! encoding: [0x9b,0xd0,0x50,0x1a]
        ! CHECK: tcc %xcc, %i5 + 41     ! encoding: [0x9b,0xd7,0x70,0x29]
        tcc %xcc, %i5
        tcc %xcc, 82
        tcc %xcc, %g1 + %i2
        tcc %xcc, %i5 + 41

        ! CHECK: tcs %xcc, %i5          ! encoding: [0x8b,0xd0,0x10,0x1d]
        ! CHECK: tcs %xcc, 82           ! encoding: [0x8b,0xd0,0x30,0x52]
        ! CHECK: tcs %xcc, %g1 + %i2    ! encoding: [0x8b,0xd0,0x50,0x1a]
        ! CHECK: tcs %xcc, %i5 + 41     ! encoding: [0x8b,0xd7,0x70,0x29]
        tcs %xcc, %i5
        tcs %xcc, 82
        tcs %xcc, %g1 + %i2
        tcs %xcc, %i5 + 41

        ! CHECK: tpos %xcc, %i5         ! encoding: [0x9d,0xd0,0x10,0x1d]
        ! CHECK: tpos %xcc, 82          ! encoding: [0x9d,0xd0,0x30,0x52]
        ! CHECK: tpos %xcc, %g1 + %i2   ! encoding: [0x9d,0xd0,0x50,0x1a]
        ! CHECK: tpos %xcc, %i5 + 41    ! encoding: [0x9d,0xd7,0x70,0x29]
        tpos %xcc, %i5
        tpos %xcc, 82
        tpos %xcc, %g1 + %i2
        tpos %xcc, %i5 + 41

        ! CHECK: tneg %xcc, %i5         ! encoding: [0x8d,0xd0,0x10,0x1d]
        ! CHECK: tneg %xcc, 82          ! encoding: [0x8d,0xd0,0x30,0x52]
        ! CHECK: tneg %xcc, %g1 + %i2   ! encoding: [0x8d,0xd0,0x50,0x1a]
        ! CHECK: tneg %xcc, %i5 + 41    ! encoding: [0x8d,0xd7,0x70,0x29]
        tneg %xcc, %i5
        tneg %xcc, 82
        tneg %xcc, %g1 + %i2
        tneg %xcc, %i5 + 41

        ! CHECK: tvc %xcc, %i5          ! encoding: [0x9f,0xd0,0x10,0x1d]
        ! CHECK: tvc %xcc, 82           ! encoding: [0x9f,0xd0,0x30,0x52]
        ! CHECK: tvc %xcc, %g1 + %i2    ! encoding: [0x9f,0xd0,0x50,0x1a]
        ! CHECK: tvc %xcc, %i5 + 41     ! encoding: [0x9f,0xd7,0x70,0x29]
        tvc %xcc, %i5
        tvc %xcc, 82
        tvc %xcc, %g1 + %i2
        tvc %xcc, %i5 + 41

        ! CHECK: tvs %xcc, %i5          ! encoding: [0x8f,0xd0,0x10,0x1d]
        ! CHECK: tvs %xcc, 82           ! encoding: [0x8f,0xd0,0x30,0x52]
        ! CHECK: tvs %xcc, %g1 + %i2    ! encoding: [0x8f,0xd0,0x50,0x1a]
        ! CHECK: tvs %xcc, %i5 + 41     ! encoding: [0x8f,0xd7,0x70,0x29]
        tvs %xcc, %i5
        tvs %xcc, 82
        tvs %xcc, %g1 + %i2
        tvs %xcc, %i5 + 41
      