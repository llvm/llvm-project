! RUN: %llvmgcc -c %s
! PR2443

! Program to test the power (**) operator
program testpow
   implicit none
   real(kind=4) r, s, two
   real(kind=8) :: q
   complex(kind=4) :: c, z
   real, parameter :: del = 0.0001
   integer i, j

   two = 2.0

   c = (2.0, 3.0)
   c = c ** two
   if (abs(c - (-5.0, 12.0)) .gt. del) call abort
end program
