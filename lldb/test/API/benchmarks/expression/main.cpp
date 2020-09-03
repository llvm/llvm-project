#include <stdio.h>

class Point {
public:
    int x;
    int y;
    Point(int a, int b):
        x(a),
        y(b)
    {}
};

class Data {
public:
    int id;
    Point point;
    Data(int i):
        id(i),
        point(0, 0)
    {}
};

int main(int argc, char const *argv[]) {
    Data *data[1000];
    Data **ptr = data;
    for (int i = 0; i < 1000; ++i) {
        ptr[i] = new Data(i);
        ptr[i]->point.x = i;
        ptr[i]->point.y = i+1;
    }

    printf("Finished populating data.\n");
    for (int j = 0; j < 1000; ++j) {
        bool dump = argc > 1; // Set breakpoint here.
                              // Evaluate a couple of expressions (2*1000 = 2000 exprs):
                              // expr ptr[j]->point.x
                              // expr ptr[j]->point.y
        if (dump) {
            printf("data[%d] = %d (%d, %d)\n", j, ptr[j]->id, ptr[j]->point.x, ptr[j]->point.y);
        }
    }
    return 0;
}
