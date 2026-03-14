#pragma once

#include <condition_variable>
#include <mutex>

int baz(int &j, std::mutex &mutex, std::condition_variable &cv);
