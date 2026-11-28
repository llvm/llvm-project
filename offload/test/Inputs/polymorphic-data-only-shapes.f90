module polymorphic_data_only_shapes
  implicit none

  type :: shape_t
    integer :: n = 0
  contains
    procedure :: bump => shape_bump
  end type shape_t

  type, extends(shape_t) :: circle_t
    integer :: r = 0
  contains
    procedure :: bump => circle_bump
  end type circle_t

contains

  subroutine shape_bump(this)
    class(shape_t), intent(inout) :: this

    this%n = this%n + 1
  end subroutine shape_bump

  subroutine circle_bump(this)
    class(circle_t), intent(inout) :: this

    this%n = this%n + 10
    this%r = this%r + 1
  end subroutine circle_bump

end module polymorphic_data_only_shapes
