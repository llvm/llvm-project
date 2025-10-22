#include <iostream>
using namespace std;

int main() {
    // 1. 基本指针操作
    int num = 42;
    int* ptr = &num; // 指针声明和初始化
    
    cout << "1. Basic pointer operations:" << endl;
    cout << "Value of num: " << num << endl;
    cout << "Address of num: " << &num << endl;
    cout << "Value of ptr: " << ptr << endl;
    cout << "Value pointed by ptr: " << *ptr << endl << endl;

    // 2. 指针解引用和修改
    *ptr = 100;
    cout << "2. After dereferencing and modification:" << endl;
    cout << "New value of num: " << num << endl << endl;

    // 3. 指针算术
    int arr[5] = {10, 20, 30, 40, 50};
    int* arrPtr = arr; // 数组名就是首元素地址
    
    cout << "3. Pointer arithmetic with array:" << endl;
    cout << "First element: " << *arrPtr << " at " << arrPtr << endl;
    arrPtr++; // 指针移动到下一个元素
    cout << "Second element: " << *arrPtr << " at " << arrPtr << endl;
    cout << "Third element: " << *(arrPtr + 1) << " at " << (arrPtr + 1) << endl << endl;

    // 4. 动态内存分配
    int* dynPtr = new int(200); // 动态分配内存
    cout << "4. Dynamic memory allocation:" << endl;
    cout << "Dynamically allocated value: " << *dynPtr << " at " << dynPtr << endl;
    
    // 5. 指针和const
    const int constNum = 300;
    const int* constPtr = &constNum; // 指向常量的指针
    cout << "5. Pointers and const:" << endl;
    cout << "Constant value through pointer: " << *constPtr << endl;
    // *constPtr = 400; // 错误：不能修改常量值
    
    // 6. 指针数组
    int* ptrArray[3] = {&num, arrPtr, dynPtr};
    cout << "6. Array of pointers:" << endl;
    for(int i = 0; i < 3; i++) {
        cout << "Pointer " << i << " points to value: " << *ptrArray[i] << endl;
    }
    cout << endl;

    // 7. 指针的比较
    cout << "7. Pointer comparison:" << endl;
    if(ptr == &num) {
        cout << "ptr is pointing to num" << endl;
    }
    if(arrPtr > arr) {
        cout << "arrPtr is ahead of arr in memory" << endl;
    }
    cout << endl;

    // 清理动态内存
    delete dynPtr;
    dynPtr = nullptr; // 避免悬空指针

    return 0;
}