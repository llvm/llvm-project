@echo off
mkdir llvm-build-debug
docker-compose up --build --remove-orphans llvm-debug
