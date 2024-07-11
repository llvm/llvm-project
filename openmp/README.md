OpenMP 是由 OpenMP Architecture Review Board 提出并广泛接受的一套指导性编译处理方案，用于共享内存并行系统的多线程程序设计。它支持的编程语言包括 C、C++ 和 Fortran，而支持 OpenMP 的编译器包括 Sun 等。OpenMP 提供对并行算法的高层抽象描述，特别适合在多核 CPU 机器上的并行程序设计。编译器根据程序中添加的 pragma 指令，自动将程序并行处理，使用 OpenMP 降低了并行编程的难度和复杂度。OpenMP 的执行模式采用 fork-join 模式。
OpenMP 的编程模型以线程为基础，通过编译制导指令（如 #pragma omp）引导并行化。它有三种编程要素可以实现并行化控制，包括编译制导、API 函数集和环境变量。编译制导指令以 #pragma omp 开始，后跟具体的功能指令，格式如：#pragma omp 指令 [子句 [,子句] …]。常用的功能指令包括 #pragma once 等，用于指定头文件只被编译一次等。
OpenMP 还利用编译器指令 (#pragma)、库函数和环境变量来简化 C/C++/Fortran 多线程编程。它不是全自动并行编程语言，其并行行为仍需由用户定义及控制。例如，#pragma GCC poison printf 用于禁止使用 printf 函数，并声明其被污染，紧跟在该条语句后面的代码段不能使用 printf。

https://www.cnblogs.com/lfri/p/10111315.html
