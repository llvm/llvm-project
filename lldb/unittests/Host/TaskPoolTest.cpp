#include "gtest/gtest.h"

#include "lldb/Host/TaskPool.h"

TEST(TaskPoolTest, AddTask) {
  auto fn = [](int x) { return x * x + 1; };

  auto f1 = TaskPool::AddTask(fn, 1);
  auto f2 = TaskPool::AddTask(fn, 2);
  auto f3 = TaskPool::AddTask(fn, 3);
  auto f4 = TaskPool::AddTask(fn, 4);

  ASSERT_EQ(10, f3.get());
  ASSERT_EQ(2, f1.get());
  ASSERT_EQ(17, f4.get());
  ASSERT_EQ(5, f2.get());
}

TEST(TaskPoolTest, RunTasks) {
  std::vector<int> r(4);

  auto fn = [](int x, int &y) { y = x * x + 1; };

  TaskPool::RunTasks([fn, &r]() { fn(1, r[0]); }, [fn, &r]() { fn(2, r[1]); },
                     [fn, &r]() { fn(3, r[2]); }, [fn, &r]() { fn(4, r[3]); });

  ASSERT_EQ(2, r[0]);
  ASSERT_EQ(5, r[1]);
  ASSERT_EQ(10, r[2]);
  ASSERT_EQ(17, r[3]);
}

TEST(TaskPoolTest, TaskMap) {
  int data[4];
  auto fn = [&data](int x) { data[x] = x * x; };

  TaskMapOverInt(0, 4, fn);

  ASSERT_EQ(data[0], 0);
  ASSERT_EQ(data[1], 1);
  ASSERT_EQ(data[2], 4);
  ASSERT_EQ(data[3], 9);
}
