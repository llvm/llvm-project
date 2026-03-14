#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

struct Shape
{
	bool dummy;
	Shape() : dummy(true) {}
};

struct Rectangle : public Shape {
    int w;
    int h;
    Rectangle(int W = 3, int H = 5) : w(W), h(H) {}
};

struct Circle : public Shape {
    int r;
    Circle(int R = 6) : r(R) {}
};

int main (int argc, const char * argv[])
{
    Rectangle r1(5,6);
    Rectangle r2(9,16);
    Rectangle r3(4,4);
    
    Circle c1(5);
    Circle c2(6);
    Circle c3(7);
    
    Circle *c_ptr = new Circle(8);
    Rectangle *r_ptr = new Rectangle(9,7);
    
    return 0; // Set break point at this line.
}

