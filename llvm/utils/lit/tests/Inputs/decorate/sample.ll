; RUN: echo opt %s -S -passes='instcombine' | echo FileCheck %s
