program frameVariable
  implicit none

  real  :: num_real
  integer :: num_int
  logical :: num_logical
  complex :: num_complex


  num_int    = 152
  num_real   = 2.718281828459045
  num_logical = .TRUE.
  num_complex = (1.3, 2.6)

  print *, "Done" ! Breakpoint here

end program frameVariable