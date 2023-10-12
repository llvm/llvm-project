#include <stdio.h>

class Task {
public:
    int id;
    Task *next;
    enum {
        TASK_TYPE_1,
        TASK_TYPE_2
    } type;
    enum E {} e;
    union U {} u;
    // This struct is anonymous b/c it does not have a name
    // and it is not unnamed class.
    // Anonymous classes are a GNU extension.
    struct {
      int y;
    };
    // This struct is an unnamed class see [class.pre]p1
    // http://eel.is/c++draft/class#pre-1.sentence-6
    struct {
      int x;
    } my_type_is_nameless;
    struct name {
      int x;
    } my_type_is_named;
    Task(int i, Task *n):
        id(i),
        next(n),
        type(TASK_TYPE_1)
    {}
};

enum EnumType {};
enum class ScopedEnumType {};
enum class EnumUChar : unsigned char {};

int main (int argc, char const *argv[])
{
    Task *task_head = new Task(-1, NULL);
    Task *task1 = new Task(1, NULL);
    Task *task2 = new Task(2, NULL);
    Task *task3 = new Task(3, NULL); // Orphaned.
    Task *task4 = new Task(4, NULL);
    Task *task5 = new Task(5, NULL);

    task_head->next = task1;
    task1->next = task2;
    task2->next = task4;
    task4->next = task5;

    int total = 0;
    Task *t = task_head;
    while (t != NULL) {
        if (t->id >= 0)
            ++total;
        t = t->next;
    }
    printf("We have a total number of %d tasks\n", total);

    // This corresponds to an empty task list.
    Task *empty_task_head = new Task(-1, NULL);

    typedef int myint;
    myint myint_arr[] = {1, 2, 3};

    EnumType enum_type;
    ScopedEnumType scoped_enum_type;
    EnumUChar scoped_enum_type_uchar;

    return 0; // Break at this line
}
