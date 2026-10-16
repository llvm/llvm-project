!===- openacc.f90 - OpenACC runtime module ---------------------------===!
!
! Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
! See https://llvm.org/LICENSE.txt for license information.
! SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
!
!===----------------------------------------------------------------------===!

module openacc
  use :: iso_c_binding, only: c_bool, c_int, c_ptr, c_size_t
  implicit none
  public

  include "AccFModule.inc"
end module
