! Offloading test checking USM with derived types containing allocatable members.
! This ensures that the 'close' map flag is not incorrectly applied to members
! when the parent is in USM, preventing runtime crashes.
! REQUIRES: flang, amdgpu

! RUN: %libomptarget-compile-fortran-generic -fopenmp-force-usm
! RUN: env LIBOMPTARGET_INFO=16 HSA_XNACK=1 %libomptarget-run-generic 2>&1 | %fcheck-generic

MODULE globalmod
   IMPLICIT NONE
   TYPE :: GRID_type
      REAL(KIND=8),ALLOCATABLE,DIMENSION(:) :: DZ
      REAL(KIND=8),ALLOCATABLE,DIMENSION(:) :: RDZ
   END TYPE GRID_type
   TYPE(GRID_type) :: GRID
END MODULE globalmod

PROGRAM reproducer
   USE globalmod
   IMPLICIT NONE

   ALLOCATE(GRID%DZ(10))
   ALLOCATE(GRID%RDZ(10))
   GRID%DZ = 1.0
   GRID%RDZ = 2.0

   !$OMP TARGET
   GRID%DZ(1) = GRID%DZ(1) + GRID%RDZ(1)
   !$OMP END TARGET

   PRINT *, GRID%DZ(1)
END PROGRAM reproducer

! CHECK: PluginInterface device {{[0-9]+}} info: Launching kernel
! CHECK: 3.
