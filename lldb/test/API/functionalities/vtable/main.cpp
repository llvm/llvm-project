class Shape {
public:
  virtual double Area() { return 1.0; }
  virtual double Perimeter() { return 1.0; }
  // Note that destructors generate two entries in the vtable: base object
  // destructor and deleting destructor.
  virtual ~Shape() = default;
};

class Rectangle : public Shape {
public:
  ~Rectangle() override = default;
  double Area() override { return 2.0; }
  double Perimeter() override { return 2.0; }
  virtual void RectangleOnly() {}
  // This *shouldn't* show up in the vtable.
  void RectangleSpecific() { return; }
};

// Make a class that looks like it would be virtual because the first ivar is
// a virtual class and if we inspect memory at the address of this class it
// would appear to be a virtual class. We need to make sure we don't get a
// valid vtable from this object.
class NotVirtual {
  Rectangle m_rect;
public:
  NotVirtual() = default;
};

int main(int argc, const char **argv) {
  Shape shape;
  Rectangle rect;
  Shape *shape_ptr = &rect;
  Shape &shape_ref = shape;
  shape_ptr = &shape; // Shape is Rectangle
  NotVirtual not_virtual; // Shape is Shape
  return 0; // At the end
}
