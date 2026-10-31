#include <memory>

class Shape {
public:
  virtual ~Shape() = default;
  int shape_val = 10;
};

class Drawable {
public:
  virtual ~Drawable() = default;
  virtual void draw() = 0;
};

class Circle : public Shape, public Drawable {
public:
  int circle_val = 20;
  void draw() override {}
};

struct Point {
  int x, y;
};

int main() {
  Point pt = { 1, 2 };
  Point points[] = {{1010,2020}, {3030,4040}, {5050,6060}};
  Point *pt_ptr = &points[1];
  Point &pt_ref = pt;
  std::shared_ptr<Point> pt_sp(new Point{111,222});

  Shape shape{};
  Shape &shape_ref = shape;
  Shape *shape_ptr = &shape;

  Circle circle{};
  Circle &circle_ref = circle;
  Circle *circle_ptr = &circle;

  Shape &circle_as_shape_ref = circle;
  Shape *circle_as_shape_ptr = &circle;

  Drawable &circle_as_drawable_ref = circle;
  Drawable *circle_as_drawable_ptr = &circle;

  return 0; // Set a breakpoint here
}

